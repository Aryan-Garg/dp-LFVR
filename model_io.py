import os

import torch
from torch.nn.parameter import Parameter


def save_weights(model, filename, path="./saved_models"):
    if not os.path.isdir(path):
        os.makedirs(path)

    fpath = os.path.join(path, filename)
    torch.save(model.state_dict(), fpath)
    return


def save_checkpoint(model, optimizer, scheduler, epoch, filename, root="./checkpoints"):
    if not os.path.isdir(root):
        os.makedirs(root)

    fpath = os.path.join(root, filename)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch
        }
        , fpath)


def load_weights(model, checkpoint):
    state_dict = model.state_dict()
    for name, param in checkpoint.items():
        if name not in state_dict:
                continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            state_dict[name].copy_(param)
        except:
            continue
    # print(f"Loaded ckpt:", checkpoint)
    return model


def load_checkpoint(fpath, model, optimizer=None, scheduler=None, replace=True, total_epochs=None, lenTrainLoader=None):
    assert total_epochs is not None, "[model_io.py] total_epochs is None"
    assert lenTrainLoader is not None, "[model_io.py] lenTrainLoader is None"
    
    ckpt = torch.load(fpath, map_location='cpu')
    if optimizer is None:
        optimizer = ckpt.get('optimizer', None)
    else:
        optimizer.load_state_dict(ckpt['optimizer'])

    epoch = ckpt['epoch'] + 1

    if scheduler is None:
        scheduler = ckpt.get('scheduler', None)
    else:
        sch_dict = ckpt['scheduler']
        # print(f"sch total_steps:",sch_dict['total_steps'])
        # exit(0)
        sch_dict['total_steps'] = sch_dict['total_steps'] + ((total_epochs - epoch) * lenTrainLoader)
        scheduler.load_state_dict(ckpt['scheduler'])
    

    if 'model' in ckpt:
        ckpt = ckpt['model']
    load_dict = {}

    for k, v in ckpt.items():
        if k.startswith('module.') and replace:
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    modified = {}  # backward compatibility to older naming of architecture blocks
    for k, v in load_dict.items():
        if k.startswith('adaptive_bins_layer.embedding_conv.'):
            k_ = k.replace('adaptive_bins_layer.embedding_conv.',
                           'adaptive_bins_layer.conv3x3.')
            modified[k_] = v
            # del load_dict[k]

        elif k.startswith('adaptive_bins_layer.patch_transformer.embedding_encoder'):

            k_ = k.replace('adaptive_bins_layer.patch_transformer.embedding_encoder',
                           'adaptive_bins_layer.patch_transformer.embedding_convPxP')
            modified[k_] = v
            # del load_dict[k]
        else:
            modified[k] = v  # else keep the original

    model.load_state_dict(modified)
    return model, optimizer, scheduler, epoch