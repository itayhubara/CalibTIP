import pandas as pd
from pulp import *
import numpy as np
import argparse
from main import main_with_args as main_per_layer

Debug = False

def mpip_compression(files=None, replace_precisions=None, Degradation=None, noise=None, method='acc', base_precision=8):

    data = {}
    for f, prec in zip(files, replace_precisions):
        data[prec] = pd.read_csv(f)

    if Degradation is None:
        Degradation = 0.18

    bops=False
    metric = 'MACs' if bops else 'Parameters Size [Elements]'

    if method=='acc':
        acc=True
    elif method=='loss':
        acc=False
    measurement = 'accuracy' if acc else 'loss'

    po = 2 if bops else 1
    prob = LpProblem('BitAllocationProblem',LpMinimize)
    Combinations={}; accLoss={}; memorySaved={}; Indicators={}; S={}; DeltaL={}

    num_layers = len(data[replace_precisions[0]]['base precision']) - 1

    base_accuracy = data[replace_precisions[0]][measurement][0]
    total_mac=0
    for l in range(1,num_layers+1):
        layer = data[replace_precisions[0]]['replaced layer'][l]
        total_mac+= int(data[replace_precisions[0]][metric][l])
        base_performance = int(data[replace_precisions[0]][metric][l]) * (base_precision ** po)
        acc_layer = {}
        performance = {}
        Combinations[layer] = []
        accLoss[layer] = {}
        memorySaved[layer] = {}
        for prec in replace_precisions:
            acc_layer[prec] = data[prec][measurement][l]
            performance[prec] = int(data[prec][metric][l]) * (prec ** po)
            Combinations[layer].append(layer + '_{}W_{}A'.format(prec, prec))
            if acc:
                accLoss[layer][layer + '_{}W_{}A'.format(prec, prec)] = max(base_accuracy - acc_layer[prec], 1e-6)
            else:
                accLoss[layer][layer + '_{}W_{}A'.format(prec, prec)] = max(acc_layer[prec] - base_accuracy, 1e-6)
            if noise is not None:
                accLoss[layer][layer + '_{}W_{}A'.format(prec, prec)] += noise * np.random.normal() * accLoss[layer][layer + '_{}W_{}A'.format(prec, prec)]
            memorySaved[layer][layer + '_{}W_{}A'.format(prec, prec)] = base_performance - performance[prec]
        Combinations[layer].append(layer + '_{}W_{}A'.format(base_precision, base_precision))
        accLoss[layer][layer + '_{}W_{}A'.format(base_precision, base_precision)] = 0
        memorySaved[layer][layer + '_{}W_{}A'.format(base_precision, base_precision)] = 0
        Indicators[layer] = LpVariable.dicts("indicator"+layer,Combinations[layer],0,1,LpInteger)
        S[layer] =LpVariable("S"+layer, 0)
        DeltaL[layer] =LpVariable("DeltaL"+layer, 0)

    prob += lpSum([S[layer] for layer in S.keys()]) # Objective (minimize acc loss)

    total_performance=total_mac*base_precision**po

    for l in range(1,num_layers+1): # range(1,3):#
        layer = data[replace_precisions[0]]['replaced layer'][l]
        prob += lpSum([Indicators[layer][i] * accLoss[layer][i] for i in Combinations[layer]]) == S[layer]  # Accuracy loss per layer
        prob += lpSum([Indicators[layer][i] for i in Combinations[layer]]) == 1  # Constraint of only one indicator==1
        prob += lpSum([Indicators[layer][i] * memorySaved[layer][i] for i in Combinations[layer]]) == DeltaL[layer]  # Acc loss per layer

    prob += lpSum([DeltaL[layer] for layer in DeltaL.keys()]) >= total_performance*(1- Degradation*(32/base_precision)) # Total acc loss constraint

    prob.solve()
    LpStatus[prob.status]

    print('optimal solution for total degradation D = ' + str(Degradation)+':')
    if Debug:
        for v in prob.variables():
            print(v.name, "=", v.varValue)

        print(value(prob.objective))

    if (prob.status==-1):
        print('Infeasable')

    expected_acc_deg = sum([S[layer].varValue for layer in S.keys()])
    reduced_performance=sum([DeltaL[layer].varValue for layer in DeltaL.keys()])

    sol = {}
    memory_reduced = 0
    acc_deg = 0
    policy = []
    all_precisions = replace_precisions + [base_precision]
    total_params = {}
    for prec in all_precisions:
        total_params[prec] = 0
    for l in range(1, num_layers + 1):
        layer = data[replace_precisions[0]]['replaced layer'][l]
        for prec in all_precisions:
            if Indicators[layer][layer + '_{}W_{}A'.format(prec, prec)].varValue:
                policy.append(prec)
                sol[layer] = [prec, prec]
                memory_reduced += memorySaved[layer][layer + '_{}W_{}A'.format(prec, prec)]
                acc_deg += accLoss[layer][layer + '_{}W_{}A'.format(prec, prec)]
                total_params[prec] += int(data[replace_precisions[0]][metric][l])

    print('Final Solution: ', sol)
    print('Policy: ', policy)
    print('Achieved compression: ', (total_performance - memory_reduced) / (total_performance * (32/base_precision)))
    if acc:
        expected_acc = base_accuracy - acc_deg
    else:
        expected_acc = base_accuracy + acc_deg
    print('Expected acc: ', expected_acc)
    for prec in all_precisions:
        print('Params % in int {} = {}'.format(prec, total_params[prec] / total_mac))

    return sol, expected_acc, (total_performance - reduced_performance) / (total_performance * (32/base_precision)), policy


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Reinforcement Learning')

    parser.add_argument('--device-ids', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3')
    parser.add_argument('--ip_method', type=str, default='loss', help='IP optimization target, loss / acc')
    parser.add_argument('--model', type=str, default='resnet', help='model to use')
    parser.add_argument('--model_vis', type=str, default='resnet50', help='torchvision model name')
    parser.add_argument('--num_exp', default=1, type=int, help='number of experiments per compression level')
    parser.add_argument('--sigma', default=None, type=float, help='sigma noise to add to measurements')
    parser.add_argument('--layer_by_layer_files', type=str, default='./results/resnet50_w8a8_adaquant/resnet.absorb_bn.measure.adaquant.per_layer_accuracy.csv', help='layer degradation csv file')
    parser.add_argument('--datasets-dir', type=str, default='/media/drive/Datasets', help='dataset dir')
    parser.add_argument('--precisions', type=str, default='8;4', help='precisions, base first, separated by ;')
    parser.add_argument('--max_compression', type=float, default='0.25', help='max compression to test')
    parser.add_argument('--min_compression', type=float, default='0.13', help='min compression to test')
    parser.add_argument('--suffix', type=str, default='', help='suffix to add to all outputs')
    parser.add_argument('--do_not_use_adaquant', action='store_true', default=False,
                        help='use non optimized model')
    parser.add_argument('--eval_on_train', action='store_true', default=False,
                        help='evaluate on calibration data')

    args = parser.parse_args()
    return args


args = get_args()

compressions = np.arange(args.min_compression, args.max_compression, 0.01)
sigma = args.sigma
num_exp = args.num_exp
ip_method = args.ip_method
files = args.layer_by_layer_files.split(';')
precisions = [int(i) for i in args.precisions.split(';')]
replace_precisions = precisions[1:]
datasets_dir = args.datasets_dir
model = args.model
model_vis = args.model_vis
if args.do_not_use_adaquant:
    workdirs = [os.path.join('results', model_vis + '_w{}a{}'.format(i, i)) for i in precisions]
else:
    workdirs = [os.path.join('results', model_vis + '_w{}a{}.adaquant'.format(i, i)) for i in precisions]
eval_dir = os.path.join(workdirs[0], model + '.absorb_bn')

perC=True
num_sp_layers=0
model_config = {'batch_norm': False,'measure': False, 'perC': perC}
if model_vis=='resnet18':
    model_config['depth'] = 18

output_fname = os.path.join(workdirs[0], 'IP_{}_{}{}.txt'.format(model_vis, ip_method, args.suffix))

eval_dict = {'model': model,
             'evaluate': eval_dir,
             'dataset': 'imagenet_calib',
             'datasets_dir': datasets_dir,
             'b': 100,
             'model_config': model_config,
             'mixed_builder': True,
             'device_ids': args.device_ids,
             'precisions': precisions}

if args.do_not_use_adaquant:
    eval_dict['opt_model_paths'] = [os.path.join(dd, model + '.absorb_bn.measure_perC') for dd in workdirs]
else:
    eval_dict['opt_model_paths'] = [os.path.join(dd, model + '.absorb_bn.measure_perC.adaquant') for dd in workdirs]

if args.eval_on_train:
    eval_dict['eval_on_train'] = True

solutions = []
expected_accuracies = []
state_dict_path=[]
actual_compressions = []
actual_accuracies = []
actual_losses = []
policies = []
completed = 0
start_from = 0
for Deg in compressions:
    if completed < start_from:
        completed += 1
        solutions.append('')
        state_dict_path.append('')
        policies.append([])
        expected_accuracies.append(0)
        actual_compressions.append(0)
        actual_accuracies.append(0)
        actual_losses.append(0)
        continue
    attempted_policies = {}
    valid_exp = 0
    while valid_exp < num_exp:
        if Debug:
            import pdb; pdb.set_trace()
        sol, expect_acc, comp, policy = mpip_compression(files=files, replace_precisions=replace_precisions, Degradation=Deg, noise=sigma, method=ip_method)
        if str(policy) in attempted_policies.keys():
            continue
        valid_exp += 1

        eval_dict['names_sp_layers'] = sol
        eval_dict['suffix'] = 'comp_{}_{}{}'.format( "{:.2f}".format(Deg), ip_method, args.suffix)
        acc, loss = main_per_layer(**eval_dict)
        # acc = 0.11; loss = 0.9
        # import pdb; pdb.set_trace()

        attempted_policies[str(policy)] = acc

        solutions.append(sol)
        policies.append(policy.copy())
        expected_accuracies.append(expect_acc)
        actual_compressions.append(comp)
        actual_accuracies.append(acc)
        actual_losses.append(loss)
        state_dict_path.append(eval_dict['evaluate']+'.mixed-ip-results.'+eval_dict['suffix'])
        completed += 1
        c = 0
        for d in compressions:
            for exp in range(num_exp):
                if c >= completed:
                    break
                print('Compression thr {}, experiment {},state_dict_path {}, compression {}, expected {} {}, actual acc {}, actual loss {}'.format("{:.2f}".format(d), exp, state_dict_path[c], actual_compressions[c],
                                                                                            ip_method, expected_accuracies[c],
                                                                                            actual_accuracies[c], actual_losses[c]))
                print('Policy: {}'.format(policies[c]))
                print('Configuration = {}'.format(solutions[c]))
                c += 1


with open(output_fname, 'w') as pid:
    line = 'Compression thr\tExperiment\tstate_dict_path\tActual compression\tExpected {}\tActual Accuracy\tActual loss\tPolicy\tConfiguration\n'.format(ip_method)
    pid.write(line)
    c = 0
    for Deg in compressions:
        for exp in range(num_exp):
            print('Compression thr {}, experiment {}, state_dict_path {}, actual_compression {}, expected {} {}, actual acc {}, actual loss {}'.format(Deg, exp,state_dict_path[c], actual_compressions[c], ip_method, expected_accuracies[c], actual_accuracies[c], actual_losses[c]))
            print('Policy: {}'.format(policies[c]))
            line = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format( "{:.2f}".format(Deg), exp, state_dict_path[c], actual_compressions[c], expected_accuracies[c], actual_accuracies[c], actual_losses[c], policies[c], solutions[c])
            pid.write(line)
            c += 1
