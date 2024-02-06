import numpy as np
import torch
import torch.nn.functional as F


"""
Implementation of rational-quadratic splines in this file is taken from
https://github.com/bayesiains/nsf.
Thank you to the authors for providing well-documented source code!
"""

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def unconstrained_RQS(inputs,
                      unnormalized_widths,
                      unnormalized_heights,
                      unnormalized_derivatives,
                      inverse=False,
                      tail_bound=1.,
                      min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                      min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                      min_derivative=DEFAULT_MIN_DERIVATIVE,
                      left=None,
                      right=None,
                      bottom=None,
                      top=None):

    if tail_bound is not None:
        bottom = -tail_bound
        top = tail_bound
        left = -tail_bound
        right = tail_bound
    if type(bottom) is not torch.Tensor:
        bottom = torch.tensor(bottom).to(inputs)
    if type(top) is not torch.Tensor:
        top = torch.tensor(top).to(inputs)
    if type(left) is not torch.Tensor:
        left = torch.tensor(left).to(inputs)
    if type(right) is not torch.Tensor:
        right = torch.tensor(right).to(inputs)

    if inverse:
        inside_intvl_mask = (inputs >= bottom) & (inputs <= top)
    else:
        inside_intvl_mask = (inputs >= left) & (inputs <= right)
    outside_interval_mask = ~inside_intvl_mask

    if type(bottom) is torch.Tensor and bottom.shape == inputs.shape:
        bottom = bottom[inside_intvl_mask]
    if type(top) is torch.Tensor and top.shape == inputs.shape:
        top = top[inside_intvl_mask]
    if type(left) is torch.Tensor and left.shape == inputs.shape:
        left = left[inside_intvl_mask]
    if type(right) is torch.Tensor and right.shape == inputs.shape:
        right = right[inside_intvl_mask]

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0

    outputs[inside_intvl_mask], logabsdet[inside_intvl_mask] = RQS(
        inputs=inputs[inside_intvl_mask],
        unnormalized_widths=unnormalized_widths[inside_intvl_mask, :],
        unnormalized_heights=unnormalized_heights[inside_intvl_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_intvl_mask, :],
        inverse=inverse,
        left=left, right=right, bottom=bottom, top=top,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )
    return outputs, logabsdet

def RQS(inputs, unnormalized_widths, unnormalized_heights,
        unnormalized_derivatives, inverse=False, left=0., right=1.,
        bottom=0., top=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE):

    if not inverse and (torch.any(inputs < left) or torch.any(inputs > right)):
        raise ValueError("Input outside domain")
    elif inverse and (torch.any(inputs < bottom) or torch.any(inputs > top)):
        raise ValueError("Input outside domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)

    if len(right.shape) != 0:
        right = right.unsqueeze(-1)
    if len(left.shape) != 0:
        left = left.unsqueeze(-1)

    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left.reshape(-1)
    cumwidths[..., -1] = right.reshape(-1) + 1e-6
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)

    if len(top.shape) != 0:
        top = top.unsqueeze(-1)
    if len(bottom.shape) != 0:
        bottom = bottom.unsqueeze(-1)

    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom.reshape(-1)
    cumheights[..., -1] = top.reshape(-1) + 1e-6
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = torch.searchsorted(cumheights, inputs.unsqueeze(-1), right=True) - 1
    else:
        bin_idx = torch.searchsorted(cumwidths, inputs.unsqueeze(-1), right=True) - 1

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)
    input_derivatives_plus_one = input_derivatives_plus_one[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives \
            + input_derivatives_plus_one - 2 * input_delta) \
            + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives - (inputs - input_cumheights) \
            * (input_derivatives + input_derivatives_plus_one \
            - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta \
                      + ((input_derivatives + input_derivatives_plus_one \
                      - 2 * input_delta) * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) \
                               * (input_derivatives_plus_one * root.pow(2) \
                                + 2 * input_delta * theta_one_minus_theta \
                                + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) \
                    + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives \
                      + input_derivatives_plus_one - 2 * input_delta) \
                      * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) \
                               * (input_derivatives_plus_one * theta.pow(2) \
                                + 2 * input_delta * theta_one_minus_theta \
                                + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet