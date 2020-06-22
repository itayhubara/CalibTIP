import torch

def correct_bias(path_bias_quant,path_bias_measure,path_model):
    model=torch.load(path_model)
    bias_quant=torch.load(path_bias_quant)
    bias_measure=torch.load(path_bias_measure)
    for key in bias_quant:
        if 'count' not in key:
            #import pdb; pdb.set_trace()
            diff_bias = (bias_quant[key]-bias_measure[key]).div(bias_measure[key+'.count']*10)
            model[key+'.bias'] -= diff_bias.to(model[key+'.bias'])
    torch.save(model,path_model+'_bias_correct')    

correct_bias('bias_mean_quant','bias_mean_measure','./results/resnet50/resnet.absorb_bn.measure_v2')