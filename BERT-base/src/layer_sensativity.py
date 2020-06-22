import torch
from torch.utils import data
import torch.nn as nn
from transformers.modeling_quantize import calculate_qparams, quantize, QConv2d, QLinear, QMatmul, QEmbedding

def search_replace_layer(model,all_names,num_bits_activation,num_bits_weight,name_model=''):
    for i,m in enumerate(model.children()):
        modules_names=[key for key in model._modules.keys()]
        layer_name=name_model+'.'+modules_names[i] if name_model !='' else name_model+modules_names[i]
        m.name=layer_name
        if layer_name in all_names:
            if isinstance(all_names,dict):
               num_bits_activation,num_bits_weight = all_names[layer_name]
            print("Layer {}, precision switch from {}-bit to {}-bit weight, {}-bit activation.".format(
                layer_name, m.num_bits, num_bits_weight, num_bits_activation))
            m.num_bits=num_bits_activation
            m.num_bits_weight = num_bits_weight
            if isinstance(m, QLinear):
                m.quantize_input.num_bits=num_bits_activation
                m.quantize_weight.num_bits=num_bits_weight
            if isinstance(m, QMatmul):
                m.quantize_input1.num_bits=num_bits_activation
                m.quantize_input2.num_bits=num_bits_activation  
            if isinstance(m, QEmbedding):
                m.quantize_weight.num_bits=num_bits_weight                   
        search_replace_layer(m,all_names,num_bits_activation,num_bits_weight,layer_name)
    return model    

def is_q_module(m):
    return isinstance(m, QConv2d) or isinstance(m, QLinear) or isinstance(m, QMatmul) or isinstance(m, QEmbedding)

def extract_all_quant_layers_names(model,q_names=[],name_model=''):
    for i,m in enumerate(model.children()):
        modules_names=[key for key in model._modules.keys()]
        layer_name=name_model+'.'+modules_names[i] if name_model !='' else name_model+modules_names[i]
        m.name=layer_name
        if is_q_module(m):
            q_names.append(m.name)
        q_names = extract_all_quant_layers_names(m,q_names,layer_name)
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
