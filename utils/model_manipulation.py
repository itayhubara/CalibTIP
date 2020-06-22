import torch
from torch.utils import data
import torch.nn as nn
from models.modules.quantize import QConv2d,QLinear

def replace_layer(model,layer,exec_str_m,new_layer_str='=QConv2d(',num_bits=8,num_bits_weight=8,forced_bias=False):
    new_layer_str='=QConv2d(' if is_Conv(layer) else '=QLinear('
    bias_str = 'True,' if forced_bias else exec_str_m+'.bias is not None,'
    #import pdb; pdb.set_trace()
    if 'Linear' in new_layer_str:
        exec_str = exec_str_m+new_layer_str + exec_str_m+'.in_features,'+exec_str_m+'.out_features,'+bias_str+'num_bits=num_bits,'+'num_bits_weight=num_bits_weight)'
    elif 'Conv' in new_layer_str:
        print(exec_str_m)
        exec_str = exec_str_m+new_layer_str+exec_str_m+'.in_channels,'+exec_str_m+'.out_channels,'+exec_str_m+'.kernel_size,'+exec_str_m+'.stride,'+exec_str_m+'.padding,'+exec_str_m+'.dilation,'+exec_str_m+'.groups,'+bias_str+'num_bits=num_bits,'+'num_bits_weight=num_bits_weight)' 
    else:
        import pdb; pdb.set_trace()    
    exec(exec_str)
    return model

def is_Linear(m):
    return  isinstance(m, nn.Linear) or isinstance(m, QLinear)

def is_Conv(m):
    return isinstance(m, nn.Conv2d) or isinstance(m, QConv2d)

def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)

def is_absorbing(m):
    return isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, QConv2d) or isinstance(m, QLinear)

def search_replace_conv_linear(model,name_model='',arr=[]):
    prev = None
    for i,m in enumerate(model.children()):
        modules_names=[key for key in model._modules.keys()]
        layer_name=name_model+'.'+modules_names[i] if name_model !='' else name_model+modules_names[i]
        exec_str_m = 'model._modules[\'%s\']'%layer_name
        if is_Conv(m) or is_Linear(m):
            model=replace_layer(model,m,exec_str_m)
        #arr= search_replace_conv_linear(m,layer_name,arr)
        prev = m
    return model 

def search_delete_bn(model,name_model='',arr=[]):
    prev = None; prev_name = None; bn_absorbed_layers=[];
    for i,m in enumerate(model.children()):
        modules_names=[key for key in model._modules.keys()]
        layer_name=name_model+'.'+modules_names[i] if name_model !='' else name_model+modules_names[i]
        exec_str_m = 'model._modules[\'%s\']'%layer_name
        if is_bn(m) and is_absorbing(prev):
            if is_Conv(prev) or is_Linear(prev):
                if prev.bias is None and prev_name is not None:
                    model = replace_layer(model,prev,prev_name,forced_bias=True)
            bn_absorbed_layers.append(layer_name)      
        prev = m
        prev_name=exec_str_m
    deps_keys = [key for key in model.deps.keys()]
    for layer_name in bn_absorbed_layers:
        model._modules.pop(layer_name) 
        pop_name=deps_keys[int(layer_name)]
        new_name = model.deps[deps_keys[int(layer_name)]][1][0]
        for key in model.deps:
            for id_name,name in enumerate(model.deps[key][1]):
                if name==pop_name:
                    model.deps[key][1][id_name]=new_name
        #import pdb; pdb.set_trace()
        model.deps.pop(deps_keys[int(layer_name)])
        #import pdb; pdb.set_trace() 
    return model
