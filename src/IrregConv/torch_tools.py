"""
PyTorch Implementation of IrregConv, by Robin Deuher
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def rand_mask(num_filters, num_channels, kernel_dims = (3, 3), weights_per_kernel = 4, center_weight = True):
    """
    args:
        num_filters:
        num_channels:
        weights_per_kernel:
    return:
        mask: a randomly generated mask for irreg conv, shape num_filters, num_channels, 3, 3
    """
    if isinstance(kernel_dims, int):
        kernel_dims = (kernel_dims, kernel_dims)

    total_weights = kernel_dims[0]*kernel_dims[1]

    if weights_per_kernel is None:
        weights_per_kernel = total_weights//2

    if kernel_dims[0] == kernel_dims[1] and kernel_dims[0]%2 == 1:
        center_weight = True 
        center_i, center_j = kernel_dims[0]//2, kernel_dims[1]//2
    else:
        center_weight = False
    
    choices = []
    for i in range(kernel_dims[0]):
        for j in range(kernel_dims[1]):
            if center_weight:
                if i != center_i or j != center_j:
                    choices.append((i, j))
            else: 
                choices.append((i, j))

    mask = np.zeros((num_filters, num_channels, kernel_dims[0], kernel_dims[1]))
    for f in range(num_filters):
        for c in range(num_channels):
            if center_weight: 
                mask[f, c, center_i, center_j] = total_weights/weights_per_kernel
            ks = np.random.permutation(len(choices))
            for k in ks[:weights_per_kernel-1]: 
                i, j = choices[k]
                mask[f, c, i, j] = total_weights/weights_per_kernel
    return torch.Tensor(mask)

class IrregConv2D(nn.Conv2d):
    """"""
    def __init__(self, in_channels, out_channels, kernel_size, mask_fn = rand_mask, weights_per_kernel = None, **kwargs):
        super(IrregConv2D, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.mask = mask_fn(out_channels, in_channels, kernel_dims = kernel_size, weights_per_kernel = weights_per_kernel)
        # get initalized weights from 
        weights = self.weight
        self.weight = nn.Parameter(weights * self.mask)

    def forward(self, input: Tensor) -> Tensor:
        weight = torch.mul(self.weight, self.mask)
        return self._conv_forward(input, weight, self.bias)
