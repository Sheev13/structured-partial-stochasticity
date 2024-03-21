import torch
from typing import Tuple, List, Optional    

def apply_mask(weights: torch.Tensor, mask: torch.Tensor, zero_magnitude: bool):
    num_samples = weights.shape[0]
    mask = mask.unsqueeze(0).repeat(num_samples, 1, 1)

    if zero_magnitude:
        return weights * mask
    else:
        weights[mask != 0.0] = mask[mask != 0.0]
        return weights


def get_mask(
    dim1: int, dim2: int, odd: bool, c: Optional[float] = None, minimal: bool = True, map_weights: Optional[torch.Tensor] = None
):
    zeros_mask = _get_zeros_mask(dim1, dim2, odd, minimal)
    if c is None and map_weights is None:
        c = 0.0
    if c == 0.0:
        return zeros_mask, zeros_mask
    else:
        if map_weights is not None:
            c = 1.0
        return _zeros_mask_to_nonzero(zeros_mask, map_weights) * c, zeros_mask


def _get_zeros_mask(dim1: int, dim2: int, odd: bool, minimal: bool):
    dim1 = dim1 - 1 # this is to ignore the bias
    mindim = min(dim1, dim2)
    if minimal:
        off_diag_zeros = _get_offset_diag_zeros(mindim)
    else:
        off_diag_zeros = _get_offset_triangular_zeros(mindim)

    if odd:
        off_diag_zeros = off_diag_zeros.flip(0)

    if dim1 > dim2:
        a = torch.ones((dim1 - dim2, dim2))
        mask = torch.cat((off_diag_zeros, a), dim=0)

    elif dim2 > dim1:
        a = torch.ones((dim1, dim2 - dim1))
        mask = torch.cat((a, off_diag_zeros), dim=1)

    else:
        mask = off_diag_zeros
        
    bias_ones = torch.ones((1, dim2))
    mask = torch.cat((bias_ones, mask), dim=0)

    return mask


def _zeros_mask_to_nonzero(mask: torch.Tensor, map_weights: Optional[torch.Tensor] = None):
    if map_weights is not None:
        nonzeros = map_weights
    else:
        nonzeros = torch.randint_like(mask, low=0, high=2)
        nonzeros[nonzeros == 0.0] = -1.0

    new_mask = torch.zeros_like(mask)
    new_mask[mask == 0.0] = nonzeros[mask == 0.0]

    return new_mask


def _get_offset_triangular_zeros(dim: int):
    return torch.tril(torch.ones((dim, dim)))


def _get_offset_diag_zeros(dim: int):
    ones = torch.ones((dim - 1,))
    return torch.ones((dim, dim)) - torch.diag(ones, diagonal=1)
