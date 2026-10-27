"""Construct a smaller dense Qwen3 MLP without modifying its source module."""
import copy
import torch


def prune_mlp(source, retained):
    """Slice gate/up rows and matching down columns; return a new frozen module.

    Supports the inspected bias-free SiLU Qwen3 MLP layout. Retained indices
    must be a nonempty, strictly increasing list/tuple of Python integers.
    A full-width identity selection is allowed for validation.
    """
    if not all(hasattr(source, key) for key in ('config', 'gate_proj', 'up_proj', 'down_proj',
                                               'hidden_size', 'intermediate_size')):
        raise TypeError('Expected a Qwen3-style MLP')
    linears = [source.gate_proj, source.up_proj, source.down_proj]
    if not all(isinstance(x, torch.nn.Linear) and x.bias is None for x in linears):
        raise ValueError('Only bias-free linear MLP projections are supported')
    hidden, width = source.hidden_size, source.intermediate_size
    if (type(hidden) is not int or type(width) is not int or min(hidden, width) < 1
            or source.config.hidden_size != hidden or source.config.intermediate_size != width
            or source.config.hidden_act != 'silu'):
        raise ValueError('Source MLP metadata is inconsistent or activation unsupported')
    for linear, shape in zip(linears, ((width, hidden), (width, hidden), (hidden, width))):
        if tuple(linear.weight.shape) != shape or (linear.out_features, linear.in_features) != shape:
            raise ValueError('Source projection dimensions are inconsistent')
    if type(retained) not in (list, tuple) or not all(type(x) is int for x in retained):
        raise TypeError('Retained indices must be a list/tuple of Python integers')
    if (not retained or len(retained) > width or list(retained) != sorted(set(retained))
            or retained[0] < 0 or retained[-1] >= width):
        raise ValueError('Retained indices must be nonempty, unique, sorted and in bounds')
    weight = linears[0].weight
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError('Expected floating-point dense source weights')
    if not all(x.weight.dtype == weight.dtype and x.weight.device == weight.device for x in linears):
        raise ValueError('Source projections must share dtype and device')
    config = copy.deepcopy(source.config)
    config.intermediate_size = len(retained)
    # Meta construction avoids allocating/initializing an unnecessary full copy.
    with torch.device('meta'):
        result = type(source)(config)
    indices = torch.tensor(retained, device=weight.device, dtype=torch.long)
    for name, dim in (('gate_proj', 0), ('up_proj', 0), ('down_proj', 1)):
        old = getattr(source, name).weight
        sliced = old.detach().index_select(dim, indices).contiguous()
        getattr(result, name).weight = torch.nn.Parameter(sliced, requires_grad=False)
    result.eval()
    result.requires_grad_(False)
    return result
