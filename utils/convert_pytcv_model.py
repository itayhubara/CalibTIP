import torch

def convert_pytcv_model(model,model_pytcv):
    sd=model.state_dict()
    sd_pytcv=model_pytcv.state_dict()
    convert_dict={}
    for key,key_pytcv in zip(sd.keys(),sd_pytcv.keys()):
        clean_key='.'.join(key.split('.')[:-1])
        clean_key_pytcv='.'.join(key_pytcv.split('.')[:-1])
        convert_dict[clean_key]=clean_key_pytcv
        if sd[key].shape != sd_pytcv[key_pytcv].shape:
            print(key,sd[key].shape,key_pytcv,sd_pytcv[key_pytcv].shape)
            import pdb; pdb.set_trace()
        else:
            sd[key].copy_(sd_pytcv[key_pytcv])                    
    return model
