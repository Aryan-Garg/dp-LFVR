import torch
from models import Gamma

def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


if __name__ == '__main__':
    model = Gamma.build(in_chans=5,
                        layers=3,
                        rank=12,
                        temporal=False,
                        dp_cross_attention=False)
    x = torch.rand(1, 5, 256, 192)
    t = None
    out, depth_planes, t1 = model(x, t)
    print('Output shapes: (decoder output, depth planes, state outputs)')
    print(out.shape, depth_planes.shape, t1)
    print(f'Number of parameters: {numel(model, only_trainable=False):,}')
    print(f'Number of trainable parameters: {numel(model, only_trainable=True):,}')
    
