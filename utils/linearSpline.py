import math

import torch
from torch.nn import functional as F
import numpy as np


def unconstrained_linear_spline(inputs, unnormalized_pdf,
                                inverse=False,
                                tail_bound=1.,
                                tails='linear'):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == 'linear':
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError('{} tails are not implemented.'.format(tails))

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = linear_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_pdf=unnormalized_pdf[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound
    )

    return outputs, logabsdet


def linear_spline(inputs, unnormalized_pdf,
                  inverse=False,
                  left=0., right=1., bottom=0., top=1.):
    """
    Reference:
    > Müller et al., Neural Importance Sampling, arXiv:1808.03856, 2018.
    """
    if not inverse and (torch.min(inputs) < left or torch.max(inputs) > right):
        raise Exception("Out of bound.")
    elif inverse and (torch.min(inputs) < bottom or torch.max(inputs) > top):
        raise Exception("Out of bound.")

    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    num_bins = unnormalized_pdf.size(-1)

    pdf = F.softmax(unnormalized_pdf, dim=-1)

    cdf = torch.cumsum(pdf, dim=-1)
    cdf[..., -1] = 1. + 1e-6
    cdf = F.pad(cdf, pad=(1, 0), mode='constant', value=0.0)

    if inverse:
        inv_bin_idx = (torch.searchsorted(cdf, inputs.unsqueeze(-1), right=True) - 1).squeeze().to(torch.int64)

        bin_boundaries = (torch.linspace(0, 1, num_bins+1)
                          .view([1] * inputs.dim() + [-1])
                          .expand(*inputs.shape, -1)).to(cdf)

        slopes = ((cdf[..., 1:] - cdf[..., :-1])
                  / (bin_boundaries[..., 1:] - bin_boundaries[..., :-1]))
        offsets = cdf[..., 1:] - slopes * bin_boundaries[..., 1:]

        inv_bin_idx = inv_bin_idx.unsqueeze(-1)
        input_slopes = slopes.gather(-1, inv_bin_idx)[..., 0]
        input_offsets = offsets.gather(-1, inv_bin_idx)[..., 0]

        outputs = (inputs - input_offsets) / input_slopes
        outputs = torch.clamp(outputs, 0, 1)

        logabsdet = -torch.log(input_slopes)
    else:
        bin_pos = inputs * num_bins

        bin_idx = torch.floor(bin_pos).long()
        bin_idx[bin_idx >= num_bins] = num_bins - 1

        alpha = bin_pos - bin_idx.float()

        input_pdfs = pdf.gather(-1, bin_idx[..., None])[..., 0]

        outputs = cdf.gather(-1, bin_idx[..., None])[..., 0]
        outputs += alpha * input_pdfs
        outputs = torch.clamp(outputs, 0, 1)

        bin_width = 1.0 / num_bins
        logabsdet = torch.log(input_pdfs) - np.log(bin_width)

    if inverse:
        outputs = outputs * (right - left) + left
        logabsdet = logabsdet - math.log(top - bottom) + math.log(right - left)
    else:
        outputs = outputs * (top - bottom) + bottom
        logabsdet = logabsdet + math.log(top - bottom) - math.log(right - left)

    return outputs, logabsdet