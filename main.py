import argparse
import os
import time
import logging
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
import torch.distributed as dist
from data import DataRegime
from utils.log import setup_logging, ResultsLog, save_checkpoint
from utils.optim import OptimRegime
from utils.cross_entropy import CrossEntropyLoss
from utils.misc import torch_dtypes
from utils.param_filter import FilterModules, is_bn

from datetime import datetime
from ast import literal_eval
from trainer import Trainer
from  utils.absorb_bn import *
from utils.adaquant import *
import torchvision
import scipy.optimize as opt
import torch.nn.functional as F
import warnings
import numpy as np
from models.modules.quantize import methods
from tqdm import tqdm
import pandas as pd
import math
import shutil
from models.modules.quantize import QParams
import ast
import ntpath


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--datasets-dir', metavar='DATASETS_DIR', default='/home/Datasets',
                    help='datasets dir')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input-size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model-config', default='',
                    help='additional architecture configuration')
parser.add_argument('--dtype', default='float',
                    help='type of tensor: ' +
                    ' | '.join(torch_dtypes.keys()) +
                    ' (default: float)')
parser.add_argument('--device', default='cuda',
                    help='device assignment ("cpu" or "cuda")')
parser.add_argument('--device-ids', default=[0], type=int, nargs='+',
                    help='device ids assignment (e.g 0 1 2 3')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='rank of distributed processes')
parser.add_argument('--dist-init', default='env://', type=str,
                    help='init used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=-1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts). -1 for unset (will start at 0)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--eval-batch-size', default=-1, type=int,
                    help='mini-batch size (default: same as training)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--label-smoothing', default=0, type=float,
                    help='label smoothing coefficient - default 0')
parser.add_argument('--mixup', default=None, type=float,
                    help='mixup alpha coefficient - default None')
parser.add_argument('--duplicates', default=1, type=int,
                    help='number of augmentations over singel example')
parser.add_argument('--chunk-batch', default=1, type=int,
                    help='chunk batch size for multiple passes (training)')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='cutout augmentations')
parser.add_argument('--autoaugment', action='store_true', default=False,
                    help='use autoaugment policies')
parser.add_argument('--grad-clip', default=-1, type=float,
                    help='maximum grad norm value, -1 for none')
parser.add_argument('--loss-scale', default=1, type=float,
                    help='loss scale for mixed precision training.')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--adapt-grad-norm', default=None, type=int,
                    help='adapt gradient scale frequency (default: None)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('--seed', default=123, type=int,
                    help='random seed (default: 123)')
parser.add_argument('-ab', '--absorb-bn', default = None,help='model to absorb bn for')
parser.add_argument('-lfv', '--load-from-vision', default = None,help='specify the required pretrainde model from torchvison')                    
parser.add_argument('--measure', dest='measure', action='store_true', default=False,
                    help='run measurment and save them') 
parser.add_argument('--prune', dest='prune', action='store_true', default=False,
                    help='run pruning') 
parser.add_argument('--fine-tune', dest='fine_tune', action='store_true', default=False,
                    help='fine tune the model - diffrent regime')  
parser.add_argument('--num-sp-layers', default=0, type=int,
                    help='number of layers that should switch precision')
parser.add_argument('--names-sp-layers', default=None, nargs='*',
                    help='names of layers that should switch precision')
parser.add_argument('--layers_precision_dict', '-lpd', default=None,
                    help='Dictionaly that describes precision for every layers') # "{'conv1': [8, 8], 'layer1.0.conv1': [8, 8], 'layer1.0.conv2': [4, 4], 'layer1.0.conv3': [4, 4], 'layer1.0.downsample.0': [8, 8], 'layer1.1.conv1': [4, 4], 'layer1.1.conv2': [4, 4], 'layer1.1.conv3': [4, 4], 'layer1.2.conv1': [4, 4], 'layer1.2.conv2': [4, 4], 'layer1.2.conv3': [4, 4], 'layer2.0.conv1': [4, 4], 'layer2.0.conv2': [2, 2], 'layer2.0.conv3': [4, 4], 'layer2.0.downsample.0': [4, 4], 'layer2.1.conv1': [2, 2], 'layer2.1.conv2': [4, 4], 'layer2.1.conv3': [4, 4], 'layer2.2.conv1': [2, 2], 'layer2.2.conv2': [2, 2], 'layer2.2.conv3': [4, 4], 'layer2.3.conv1': [2, 2], 'layer2.3.conv2': [2, 2], 'layer2.3.conv3': [4, 4], 'layer3.0.conv1': [4, 4], 'layer3.0.conv2': [2, 2], 'layer3.0.conv3': [2, 2], 'layer3.0.downsample.0': [2, 2], 'layer3.1.conv1': [2, 2], 'layer3.1.conv2': [2, 2], 'layer3.1.conv3': [2, 2], 'layer3.2.conv1': [2, 2], 'layer3.2.conv2': [2, 2], 'layer3.2.conv3': [2, 2], 'layer3.3.conv1': [2, 2], 'layer3.3.conv2': [2, 2], 'layer3.3.conv3': [2, 2], 'layer3.4.conv1': [2, 2], 'layer3.4.conv2': [2, 2], 'layer3.4.conv3': [2, 2], 'layer3.5.conv1': [2, 2], 'layer3.5.conv2': [2, 2], 'layer3.5.conv3': [2, 2], 'layer4.0.conv1': [2, 2], 'layer4.0.conv2': [2, 2], 'layer4.0.conv3': [2, 2], 'layer4.0.downsample.0': [2, 2], 'layer4.1.conv1': [2, 2], 'layer4.1.conv2': [2, 2], 'layer4.1.conv3': [2, 2], 'layer4.2.conv1': [2, 2], 'layer4.2.conv2': [2, 2], 'layer4.2.conv3': [2, 2], 'fc': [4, 4]}"
parser.add_argument('--keep_first_last', dest='keep_first_last', action='store_true', default=False,
                    help='keep first and last layer in base precision')     
parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=False,
                    help='loading pretrained model')  
parser.add_argument('--extract-bias-mean', dest='extract_bias_mean', action='store_true', default=False,
                    help='extract activation bias mean for correction') 
parser.add_argument('--lapq', dest='lapq', action='store_true', default=False,
                    help='lapq cliping search')    
parser.add_argument('--update_only_th', dest='update_only_th', action='store_true', default=False,
                    help='update_only_th')                        
parser.add_argument('--rec', dest='rec', action='store_true', default=False,
                    help='record output for KD learning')
parser.add_argument('--kld_loss', dest='kld_loss', action='store_true', default=False,
                    help='use kld_loss')   
parser.add_argument('--evaluate_init_configuration', dest='evaluate_init_configuration', action='store_true', default=False,
                    help='evaluate the init configuration')  
parser.add_argument('--nbits_weight', default=4, type=int,help='switch layers weights precision')
parser.add_argument('--nbits_act', default=4, type=int, help='switch layers activation precision')  
parser.add_argument('--optimize_rounding', dest='optimize_rounding', action='store_true', default=False,
                    help='optimize_rounding')
parser.add_argument('--ignore-downsample', action='store_true', default=False,
                    help='Quantize downsample layers with 8 bit')
parser.add_argument('--optimize-weights', dest='optimize_weights', action='store_true', default=False,
                    help='Optimize weights')
parser.add_argument('-i8m', '--int8_opt_model_path', type=str, metavar='FILE',
                    help='path to adaquant int8 optimized model path')   
parser.add_argument('-i4m', '--int4_opt_model_path', type=str, metavar='FILE',
                    help='path to adaquant int4 optimized model path')
parser.add_argument('-i2m', '--int2_opt_model_path', type=str, metavar='FILE',
                    help='path to adaquant int2 optimized model path')
parser.add_argument('--mixed_builder', action='store_true', default=False,
                    help='evaluate mixed precision configuration')
parser.add_argument('--suffix', default='', type=str,
                    help='suffix to add to saved mixed-ip results')                    
parser.add_argument('--adaquant', action='store_true', default=False,
                    help='Applying Adaquant MSE minimization')       
parser.add_argument('--per-layer', action='store_true', default=False,
                    help='Applying per-layer for IP mixed-precision')       
parser.add_argument('--mixed-builder', action='store_true', default=False,
                    help='evaluate IP mixed-precision results')    
parser.add_argument('--batch-norn-tuning', action='store_true', default=False,
                    help='Applying BNT method')       
parser.add_argument('--bias-tuning', action='store_true', default=False,
                    help='Applying BT method')
parser.add_argument('--eval_on_train', action='store_true', default=False,
                    help='evaluate on train set')
parser.add_argument('--opt_model_paths', type=str,
                    help='paths to all optimized models, separated by ;')
parser.add_argument('--precisions', type=str, default='4;8',
                    help='precisions of all optimized models, separated by ;')
parser.add_argument('--tuning-iter', default=1, type=int, help='Number of iterations to tune batch normalization layers.')
parser.add_argument('--res-log', default=None, help='path to result pandas log file')

def main():
    args = parser.parse_args()
    main_worker(args)

def main_with_args(**kwargs):
    args = parser.parse_known_args()[0]
    for k in kwargs.keys():
        setattr(args, k, kwargs[k])
    # args.names_sp_layers = args.names_sp_layers.split(" ")
    acc, loss = main_worker(args)
    return acc, loss
    
def main_worker(args):
    global best_prec1, dtype
    acc = -1
    loss = -1
    best_prec1 = 0
    dtype = torch_dtypes.get(args.dtype)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = time_stamp
    save_path = os.path.join(args.results_dir, args.save)

    args.distributed = args.local_rank >= 0 or args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_init,
                                world_size=args.world_size, rank=args.local_rank)
        args.local_rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        if args.dist_backend == 'mpi':
            # If using MPI, select all visible devices
            args.device_ids = list(range(torch.cuda.device_count()))
        else:
            args.device_ids = [args.local_rank]

    if not os.path.exists(save_path) and not (args.distributed and args.local_rank > 0):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'),
                  resume=args.resume is not '',
                  dummy=args.distributed and args.local_rank > 0)

    results_path = os.path.join(save_path, 'results')
    results = ResultsLog(
        results_path, title='Training Results - %s' % args.save)

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)
    logging.info("creating model %s", args.model)

    if 'cuda' in args.device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.device_ids[0])
        cudnn.benchmark = True
    else:
        args.device_ids = None

    # create model
    model = models.__dict__[args.model]
    dataset_type = 'imagenet' if args.dataset =='imagenet_calib' else args.dataset
    model_config = {'dataset': dataset_type}

    if args.model_config is not '':
        if isinstance(args.model_config, dict):
            for k, v in args.model_config.items():
                if k not in model_config.keys():
                    model_config[k] = v
        else:
            args_dict = literal_eval(args.model_config)
            for k, v in args_dict.items():
                model_config[k] = v
    if (args.absorb_bn or args.load_from_vision or args.pretrained) and not args.batch_norn_tuning:
        if args.load_from_vision:
            import torchvision
            exec_lfv_str = 'torchvision.models.' + args.load_from_vision + '(pretrained=True)'
            model = eval(exec_lfv_str)
        else:
            if not os.path.isfile(args.absorb_bn):
                parser.error('invalid checkpoint: {}'.format(args.evaluate))
            model = model_(**model_config)
            checkpoint = torch.load(args.absorb_bn,map_location=lambda storage, loc: storage)
            checkpoint = checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint
            model.load_state_dict(checkpoint,strict=False)
        if 'batch_norm' in model_config and not model_config['batch_norm']:
            logging.info('Creating absorb_bn state dict')
            search_absorbe_bn(model)
            filename_ab = args.absorb_bn+'.absorb_bn' if args.absorb_bn else save_path+'/'+args.model+'.absorb_bn'
            torch.save(model.state_dict(),filename_ab)
        else:    
            filename_bn = save_path+'/'+args.model+'.with_bn'
            torch.save(model.state_dict(),filename_bn)
        if args.load_from_vision: return
           
    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)
    
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate, map_location="cpu")
        # Overrride configuration with checkpoint info
        args.model = checkpoint.get('model', args.model)
        args.model_config = checkpoint.get('config', args.model_config)
        if not model_config['batch_norm']:
            search_absorbe_fake_bn(model)
        # load checkpoint
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s'", args.evaluate)
        else:
            model.load_state_dict(checkpoint,strict=False)    
            logging.info("loaded checkpoint '%s'",args.evaluate)
          

    if args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            if args.start_epoch < 0:  # not explicitly set
                args.start_epoch = checkpoint['epoch'] - 1 if 'epoch' in checkpoint.keys() else 0    
            best_prec1 = checkpoint['best_prec1'] if 'best_prec1' in checkpoint.keys() else -1
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint
            model.load_state_dict(sd,strict=False)
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, args.start_epoch)
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    # define loss function (criterion) and optimizer
    loss_params = {}
    if args.label_smoothing > 0:
        loss_params['smooth_eps'] = args.label_smoothing
    criterion = getattr(model, 'criterion', CrossEntropyLoss)(**loss_params)
    if args.kld_loss:
       criterion = nn.KLDivLoss(reduction='mean') 
    criterion.to(args.device, dtype)
    model.to(args.device, dtype)

    # Batch-norm should always be done in float
    if 'half' in args.dtype:
        FilterModules(model, module=is_bn).to(dtype=torch.float)

    # optimizer configuration
    optim_regime = getattr(model, 'regime', [{'epoch': 0,
                                              'optimizer': args.optimizer,
                                              'lr': args.lr,
                                              'momentum': args.momentum,
                                              'weight_decay': args.weight_decay}])
    if args.fine_tune or args.prune: 
        if not args.resume: args.start_epoch=0  
        if args.update_only_th:
            #optim_regime = [
            #    {'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-4}] 
            optim_regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1},
                {'epoch': 10, 'lr': 1e-2},
                {'epoch': 15, 'lr': 1e-3}]
        else:              
            optim_regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-4, 'momentum': 0.9},
                {'epoch': 2, 'lr': 1e-5, 'momentum': 0.9},
                {'epoch': 10, 'lr': 1e-6, 'momentum': 0.9}]
    optimizer = optim_regime if isinstance(optim_regime, OptimRegime) \
        else OptimRegime(model, optim_regime, use_float_copy='half' in args.dtype)

    # Training Data loading code
    
    train_data = DataRegime(getattr(model, 'data_regime', None),
                            defaults={'datasets_path': args.datasets_dir, 'name': args.dataset, 'split': 'train', 'augment': False,
                                      'input_size': args.input_size,  'batch_size': args.batch_size, 'shuffle': True,
                                      'num_workers': args.workers, 'pin_memory': True, 'drop_last': True,
                                      'distributed': args.distributed, 'duplicates': args.duplicates, 'autoaugment': args.autoaugment,
                                      'cutout': {'holes': 1, 'length': 16} if args.cutout else None})
    if args.names_sp_layers is None and args.layers_precision_dict is None:
        args.names_sp_layers =  [key[:-7] for key in model.state_dict().keys() if 'weight' in key and 'running' not in key and ('conv' in key or 'downsample.0' in key or 'fc' in key)]
        if args.keep_first_last: args.names_sp_layers=[name for name in args.names_sp_layers if name!='conv1' and name!='fc']
        args.names_sp_layers = [k for k in args.names_sp_layers if 'downsample' not in k] if args.ignore_downsample else args.names_sp_layers
        if args.num_sp_layers == 0 and not args.keep_first_last:
            args.names_sp_layers = []

    prunner = None 
    trainer = Trainer(model,prunner, criterion, optimizer,
                      device_ids=args.device_ids, device=args.device, dtype=dtype,
                      distributed=args.distributed, local_rank=args.local_rank, mixup=args.mixup, loss_scale=args.loss_scale,
                      grad_clip=args.grad_clip, print_freq=args.print_freq, adapt_grad_norm=args.adapt_grad_norm,epoch=args.start_epoch,update_only_th=args.update_only_th,optimize_rounding=args.optimize_rounding)

    
    # Evaluation Data loading code
    args.eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else args.batch_size     
    dataset_type = 'imagenet' if args.dataset =='imagenet_calib' else args.dataset
    val_data = DataRegime(getattr(model, 'data_eval_regime', None),
                          defaults={'datasets_path': args.datasets_dir, 'name': dataset_type, 'split': 'val', 'augment': False,
                                    'input_size': args.input_size, 'batch_size': args.eval_batch_size, 'shuffle': True,
                                    'num_workers': args.workers, 'pin_memory': True, 'drop_last': False})

    if args.evaluate or args.resume:
        from utils.layer_sensativity import search_replace_layer , extract_save_quant_state_dict, search_replace_layer_from_dict
        if args.layers_precision_dict is not None:
            model = search_replace_layer_from_dict(model, ast.literal_eval(args.layers_precision_dict))
        else:
            model = search_replace_layer(model, args.names_sp_layers, num_bits_activation=args.nbits_act,
                                         num_bits_weight=args.nbits_weight)

    cached_input_output = {}
    quant_keys = ['.weight', '.bias', '.equ_scale', '.quantize_input.running_zero_point', '.quantize_input.running_range',
         '.quantize_weight.running_zero_point', '.quantize_weight.running_range','.quantize_input1.running_zero_point', '.quantize_input1.running_range'
         '.quantize_input2.running_zero_point', '.quantize_input2.running_range']        
    if args.adaquant:
        def hook(module, input, output):
            if module not in cached_input_output:
                cached_input_output[module] = []
            # Meanwhile store data in the RAM.
            cached_input_output[module].append((input[0].detach().cpu(), output.detach().cpu()))
            print(module.__str__()[:70])

        from models.modules.quantize import QConv2d, QLinear
        handlers = []
        count = 0
        for m in model.modules():
            if isinstance(m, QConv2d) or isinstance(m, QLinear):
            # if isinstance(m, QConv2d):
                m.quantize = False
                if count < 1000:
                # if (isinstance(m, QConv2d) and m.groups == 1) or isinstance(m, QLinear):
                    handlers.append(m.register_forward_hook(hook))
                    count += 1

        # Store input/output for all quantizable layers
        trainer.validate(train_data.get_loader())
        print("Input/outputs cached")

        for handler in handlers:
            handler.remove()

        for m in model.modules():
            if isinstance(m, QConv2d) or isinstance(m, QLinear):
                m.quantize = True

        mse_df = pd.DataFrame(index=np.arange(len(cached_input_output)), columns=['name', 'bit', 'shape', 'mse_before', 'mse_after'])
        print_freq = 100
        for i, layer in enumerate(cached_input_output):
            print("\nOptimize {}:{} for {} bit of shape {}".format(i, layer.name, layer.num_bits, layer.weight.shape))
            mse_before, mse_after, snr_before, snr_after, kurt_in, kurt_w = \
                optimize_layer(layer, cached_input_output[layer], args.optimize_weights)
            print("\nMSE before optimization: {}".format(mse_before))
            print("MSE after optimization:  {}".format(mse_after))
            mse_df.loc[i, 'name'] = layer.name
            mse_df.loc[i, 'bit'] = layer.num_bits
            mse_df.loc[i, 'shape'] = str(layer.weight.shape)
            mse_df.loc[i, 'mse_before'] = mse_before
            mse_df.loc[i, 'mse_after'] = mse_after
            mse_df.loc[i, 'snr_before'] = snr_before
            mse_df.loc[i, 'snr_after'] = snr_after
            mse_df.loc[i, 'kurt_in'] = kurt_in
            mse_df.loc[i, 'kurt_w'] = kurt_w
            if i > 0 and i % print_freq == 0:
                print('\n')
                val_results = trainer.validate(val_data.get_loader())
                logging.info(val_results)

        mse_csv = args.evaluate + '.mse.csv'
        mse_df.to_csv(mse_csv)

        filename = args.evaluate + '.adaquant'
        torch.save(model.state_dict(), filename)

        train_data = None
        cached_input_output = None
        val_results = trainer.validate(val_data.get_loader())
        logging.info(val_results)
    elif args.per_layer:
        # Store input/output for all quantizable layers
        calib_all_8_results = trainer.validate(train_data.get_loader())
        print('########## All 8bit results ###########', calib_all_8_results)
        int8_opt_model_state_dict = torch.load(args.int8_opt_model_path)
        int4_opt_model_state_dict = torch.load(args.int4_opt_model_path)
        
        per_layer_results={}
        args.names_sp_layers =  [key[:-7] for key in model.state_dict().keys() if 'weight' in key and 'running' not in key and 'quantize' not in key and ('conv' in key or 'downsample.0' in key or 'fc' in key)]
        for layer_idx,layer in enumerate(args.names_sp_layers):
            model.load_state_dict(int8_opt_model_state_dict,strict=False)
            model = search_replace_layer(model,[layer],num_bits_activation=args.nbits_act,num_bits_weight=args.nbits_weight)
            layer_keys = [key for key in int8_opt_model_state_dict for qpkey in quant_keys if layer+qpkey==key]
            for key in layer_keys:
                model.state_dict()[key].copy_(int4_opt_model_state_dict[key])
            calib_results = trainer.validate(train_data.get_loader())
            model = search_replace_layer(model, [layer], num_bits_activation=8, num_bits_weight=8)
            print('finished %d out of %d'%(layer_idx,len(args.names_sp_layers)))
            logging.info(layer)
            logging.info(calib_results)
            per_layer_results[layer] = {'base precision': 8, 'replaced precision': args.nbits_act, 'replaced layer': layer, 'accuracy': calib_results['prec1'] , 'loss': calib_results['loss'], 'Parameters Size [Elements]':  model.state_dict()[layer+'.weight'].numel() , 'MACs': '-'}
        
        torch.save(per_layer_results,args.evaluate+'.per_layer_accuracy.A'+str(args.nbits_act)+'.W'+str(args.nbits_weight))
        all_8_dict = {'base precision': 8, 'replaced precision': args.nbits_act, 'replaced layer': '-', 'accuracy': calib_all_8_results['prec1'] , 'loss': calib_all_8_results['loss'], 'Parameters Size [Elements]':  '-', 'MACs': '-'}
        columns = [key for key in all_8_dict]
        with open(args.evaluate+'.per_layer_accuracy.A'+str(args.nbits_act)+'.W'+str(args.nbits_weight)+'.csv', "w") as f:
            f.write(",".join(columns) + "\n")
            col = [str(all_8_dict[c]) for c in all_8_dict.keys()]
            f.write(",".join(col) + "\n")
            for layer in per_layer_results:
                r = per_layer_results[layer]
                col = [str(r[c]) for c in r.keys()]
                f.write(",".join(col) + "\n")    
    elif args.mixed_builder:
        if isinstance(args.names_sp_layers, list):
            print('loading int8 model" ', args.int8_opt_model_path)
            int8_opt_model_state_dict = torch.load(args.int8_opt_model_path)
            print('loading int4 model" ', args.int4_opt_model_path)
            int4_opt_model_state_dict = torch.load(args.int4_opt_model_path)

            model.load_state_dict(int8_opt_model_state_dict, strict=False)
            model = search_replace_layer(model, args.names_sp_layers, num_bits_activation=args.nbits_act,
                                         num_bits_weight=args.nbits_weight)
            for layer_idx, layer in enumerate(args.names_sp_layers):
                layer_keys = [key for key in int8_opt_model_state_dict for qpkey in quant_keys if
                              layer + qpkey == key]
                for key in layer_keys:
                    model.state_dict()[key].copy_(int4_opt_model_state_dict[key])
                print('switched layer %s to 4 bit' % (layer))
        elif isinstance(args.names_sp_layers, dict):
            quant_models = {}
            base_precision = args.precisions[0]
            for m, prec in zip(args.opt_model_paths, args.precisions):
                print('For precision={}, loading {}'.format(prec, m))
                quant_models[prec] = torch.load(m)
            model.load_state_dict(quant_models[base_precision], strict=False)
            for layer_name, nbits_list in args.names_sp_layers.items():
                model = search_replace_layer(model, [layer_name], num_bits_activation=nbits_list[0],
                                             num_bits_weight=nbits_list[0])
                layer_keys = [key for key in quant_models[base_precision] for qpkey in quant_keys if
                              layer_name + qpkey == key]
                for key in layer_keys:
                    model.state_dict()[key].copy_(quant_models[nbits_list[0]][key])
                print('switched layer {} to {} bit'.format(layer_name, nbits_list[0]))
        if os.environ.get('DEBUG')=='True':
            from utils.layer_sensativity import check_quantized_model
            fp_names = check_quantized_model(trainer.model)
            if len(fp_names)>0:
                logging.info('Found FP32 layers in the model:')  
                logging.info(fp_names)
        if args.eval_on_train:
            mixedIP_results = trainer.validate(train_data.get_loader())
        else:
            mixedIP_results = trainer.validate(val_data.get_loader())
        torch.save({'state_dict': model.state_dict(), 'config-ip': args.names_sp_layers},args.evaluate+'.mixed-ip-results.'+args.suffix)
        logging.info(mixedIP_results)
        acc = mixedIP_results['prec1']
        loss = mixedIP_results['loss']
    elif args.batch_norn_tuning:
        from utils.layer_sensativity import search_replace_layer , extract_save_quant_state_dict, search_replace_layer_from_dict
        from models.modules.quantize import QConv2d
        if args.layers_precision_dict is not None:
            model = search_replace_layer_from_dict(model, literal_eval(args.layers_precision_dict))
        else:
            model = search_replace_layer(model, args.names_sp_layers, num_bits_activation=args.nbits_act,
                                         num_bits_weight=args.nbits_weight)

        exec_lfv_str = 'torchvision.models.' + args.load_from_vision + '(pretrained=True)'
        model_orig = eval(exec_lfv_str)
        model_orig.to(args.device, dtype)
        search_copy_bn_params(model_orig)
    
        layers_orig = dict([(n, m) for n, m in model_orig.named_modules() if isinstance(m, nn.Conv2d)])
        layers_q = dict([(n, m) for n, m in model.named_modules() if isinstance(m, QConv2d)])
        for l in layers_orig:
            conv_orig = layers_orig[l]
            conv_q = layers_q[l]
            conv_q.register_parameter('gamma', nn.Parameter(conv_orig.gamma.clone()))
            conv_q.register_parameter('beta', nn.Parameter(conv_orig.beta.clone()))
    
        del model_orig
    
        search_add_bn(model)
    
        print("Run BN tuning")
        for tt in range(args.tuning_iter):
            print(tt)
            trainer.cal_bn_stats(train_data.get_loader())
    
        search_absorbe_tuning_bn(model)
    
        filename = args.evaluate + '.bn_tuning'
        print("Save model to: {}".format(filename))
        torch.save(model.state_dict(), filename)
    
        val_results = trainer.validate(val_data.get_loader())
        logging.info(val_results)
    
        if args.res_log is not None:
            if not os.path.exists(args.res_log):
                df = pd.DataFrame()
            else:
                df = pd.read_csv(args.res_log, index_col=0)
    
            ckp = ntpath.basename(args.evaluate)
            df.loc[ckp, 'acc_bn_tuning'] = val_results['prec1']
            df.loc[ckp, 'loss_bn_tuning'] = val_results['loss']
            df.to_csv(args.res_log)
            # print(df)

    elif args.bias_tuning:
        for epoch in range(args.epochs):
            trainer.epoch = epoch
            train_data.set_epoch(epoch)
            val_data.set_epoch(epoch)
            logging.info('\nStarting Epoch: {0}\n'.format(epoch + 1))
            # train for one epoch
            repeat_train = 20 if args.update_only_th else 1
            for tt in range(repeat_train):
                print(tt)
                train_results = trainer.train(train_data.get_loader(),
                                              duplicates=train_data.get('duplicates'),
                                              chunk_batch=args.chunk_batch)
                logging.info(train_results)

        val_results = trainer.validate(val_data.get_loader())
        logging.info(val_results)
        if args.res_log is not None:
            if not os.path.exists(args.res_log):
                df = pd.DataFrame()
            else:
                df = pd.read_csv(args.res_log, index_col=0)

            ckp = ntpath.basename(args.resume)
            if 'bn_tuning' in ckp:
                ckp = ckp.replace('.bn_tuning', '')
            df.loc[ckp, 'acc_bias_tuning'] = best_prec1
            df.to_csv(args.res_log)
        # import pdb; pdb.set_trace()
    else:
        #print('Please Choose one of the following ....')
        if model_config['measure']:
            results = trainer.validate(train_data.get_loader(),rec=args.rec)
        else: 
            if args.evaluate_init_configuration:   
                results = trainer.validate(val_data.get_loader())
                if args.res_log is not None:
                    if not os.path.exists(args.res_log):
                        df = pd.DataFrame()
                    else:
                        df = pd.read_csv(args.res_log, index_col=0)

                    ckp = ntpath.basename(args.evaluate)
                    df.loc[ckp, 'acc_base'] = results['prec1']
                    df.loc[ckp, 'loss_base'] = results['loss']
                    df.to_csv(args.res_log)
           
        if args.extract_bias_mean:
            file_name  = 'bias_mean_measure' if model_config['measure'] else  'bias_mean_quant'
            torch.save(trainer.bias_mean,file_name)
        if model_config['measure']:
            filename = args.evaluate+'.measure'
            if 'perC' in args.model_config: filename += '_perC'
            torch.save(model.state_dict(),filename)
            logging.info(results)
        else:
            if args.evaluate_init_configuration:
                logging.info(results)
    return acc, loss
if __name__ == '__main__':
    main()
