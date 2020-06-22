import torch
from torch.utils import data
import torch.nn as nn
from models.modules.quantize import calculate_qparams, quantize, QConv2d,QLinear


def search_replace_layer(model,all_names,num_bits_activation,num_bits_weight,name_model=''):
    for i,m in enumerate(model.children()):
        modules_names=[key for key in model._modules.keys()]
        layer_name=name_model+'.'+modules_names[i] if name_model !='' else name_model+modules_names[i]
        m.name=layer_name
        if layer_name in all_names:
            print("Layer {}, precision switch from {}-bit to {}-bit weight, {}-bit activation.".format(
                layer_name, m.num_bits, num_bits_weight, num_bits_activation))
            m.num_bits=num_bits_activation
            m.num_bits_weight = num_bits_weight
            m.quantize_input.num_bits=num_bits_activation
            m.quantize_weight.num_bits=num_bits_weight
        search_replace_layer(m,all_names,num_bits_activation,num_bits_weight,layer_name)
    return model


# "{'conv1': [8, 8], 'layer1.0.conv1': [8, 8], 'layer1.0.conv2': [4, 4], 'layer1.0.conv3': [4, 4], 'layer1.0.downsample.0': [8, 8], 'layer1.1.conv1': [4, 4], 'layer1.1.conv2': [4, 4], 'layer1.1.conv3': [4, 4], 'layer1.2.conv1': [4, 4], 'layer1.2.conv2': [4, 4], 'layer1.2.conv3': [4, 4], 'layer2.0.conv1': [4, 4], 'layer2.0.conv2': [2, 2], 'layer2.0.conv3': [4, 4], 'layer2.0.downsample.0': [4, 4], 'layer2.1.conv1': [2, 2], 'layer2.1.conv2': [4, 4], 'layer2.1.conv3': [4, 4], 'layer2.2.conv1': [2, 2], 'layer2.2.conv2': [2, 2], 'layer2.2.conv3': [4, 4], 'layer2.3.conv1': [2, 2], 'layer2.3.conv2': [2, 2], 'layer2.3.conv3': [4, 4], 'layer3.0.conv1': [4, 4], 'layer3.0.conv2': [2, 2], 'layer3.0.conv3': [2, 2], 'layer3.0.downsample.0': [2, 2], 'layer3.1.conv1': [2, 2], 'layer3.1.conv2': [2, 2], 'layer3.1.conv3': [2, 2], 'layer3.2.conv1': [2, 2], 'layer3.2.conv2': [2, 2], 'layer3.2.conv3': [2, 2], 'layer3.3.conv1': [2, 2], 'layer3.3.conv2': [2, 2], 'layer3.3.conv3': [2, 2], 'layer3.4.conv1': [2, 2], 'layer3.4.conv2': [2, 2], 'layer3.4.conv3': [2, 2], 'layer3.5.conv1': [2, 2], 'layer3.5.conv2': [2, 2], 'layer3.5.conv3': [2, 2], 'layer4.0.conv1': [2, 2], 'layer4.0.conv2': [2, 2], 'layer4.0.conv3': [2, 2], 'layer4.0.downsample.0': [2, 2], 'layer4.1.conv1': [2, 2], 'layer4.1.conv2': [2, 2], 'layer4.1.conv3': [2, 2], 'layer4.2.conv1': [2, 2], 'layer4.2.conv2': [2, 2], 'layer4.2.conv3': [2, 2], 'fc': [4, 4]}"
def search_replace_layer_from_dict(model, layers_precision_dict, name_model=''):
    for i,m in enumerate(model.children()):
        modules_names=[key for key in model._modules.keys()]
        layer_name=name_model+'.'+modules_names[i] if name_model !='' else name_model+modules_names[i]
        m.name=layer_name
        if layer_name in layers_precision_dict:
            new_prec = layers_precision_dict[layer_name]
            print("Layer {}, precision switch from {}-bit to {}-bit weight, {}-bit activation.".format(
                layer_name, m.num_bits, new_prec[0], new_prec[1]))
            m.num_bits=new_prec[0]
            m.num_bits_weight = new_prec[1]
            m.quantize_input.num_bits=new_prec[1]
            m.quantize_weight.num_bits=new_prec[0]
        search_replace_layer_from_dict(m,layers_precision_dict,layer_name)
    return model


def is_q_module(m):
    return isinstance(m, QConv2d) or isinstance(m, QLinear)

def extract_all_quant_layers_names(model,q_names=[],name_model=''):
    for i,m in enumerate(model.children()):
        modules_names=[key for key in model._modules.keys()]
        layer_name=name_model+'.'+modules_names[i] if name_model !='' else name_model+modules_names[i]
        m.name=layer_name
        if is_q_module(m):
            q_names.append(m.name)
            print("Layer {}, if in FP32.".format(layer_name))
        q_names = check_quantized_model(m,q_names,layer_name)
    return q_names  
    
def check_quantized_model(model,fp_names=[],name_model=''):
    for i,m in enumerate(model.children()):
        modules_names=[key for key in model._modules.keys()]
        layer_name=name_model+'.'+modules_names[i] if name_model !='' else name_model+modules_names[i]
        m.name=layer_name
        if (is_q_module(m) and m.measure) or not is_q_module:
            fp_names.append(m.name)
            print("Layer {}, if in FP32.".format(layer_name))
        fp_names = check_quantized_model(m,fp_names,layer_name)
    return fp_names  

def extract_save_quant_state_dict(model,all_names,filename='int_state_dict.pth.tar'):
    
    state_dict=model.state_dict()
    
    for key in state_dict.keys():
        #import pdb; pdb.set_trace()
        val=state_dict[key]
        if 'weight' in key:
            num_bits = 4 if key[:-7] in all_names else 8
            if num_bits==4:
                import pdb; pdb.set_trace()
            weight_qparams = calculate_qparams(val, num_bits=num_bits, flatten_dims=(1, -1), reduce_dim=None)
            val_q=quantize(val, qparams=weight_qparams,dequantize=False) 
            zero_point=(-weight_qparams[1]/weight_qparams[0]*(2**weight_qparams[2]-1)).round()
            val_q=val_q-zero_point
            print(val_q.eq(0).sum().float().div(val_q.numel()))
        if 'bias' in key:
            val_q = quantize(val, num_bits=num_bits*2,flatten_dims=(0, -1))  

        state_dict[key] = val_q
    torch.save(state_dict,filename)        
    return state_dict  
