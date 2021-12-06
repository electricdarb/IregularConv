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

    mask = np.zeros((kernel_dims[0], kernel_dims[1], num_channels, num_filters))

    mask = np.zeros((num_filters, num_channels, kernel_dims[0], kernel_dims[1]))
    for f in range(num_filters):
        for c in range(num_channels):
            mask[1, 1, c, f] = total_weights/weights_per_kernel
            ks = np.random.permutation(len(choices))
            for k in ks[:weights_per_kernel-1]: 
                i, j = choices[k]
                mask[f, c, i, j] = total_weights/weights_per_kernel
    return torch.from_numpy(mask)

class BrainDamage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, mask, padding=0, stride=1, groups=1, dilation=1):
        if torch.cuda.is_available():
            mask = mask.to(input.device)
        if mask.size() == weights.size():
            weights = weights * mask
        output = F.conv2d(input, weights, stride=stride, padding=padding, groups=groups, dilation=dilation)
        ctx.save_for_backward(input, weights, mask, torch.tensor(stride), torch.tensor(padding), torch.tensor(groups), torch.tensor(dilation))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, w, mask, stride, padding, groups, dilation = ctx.saved_variables
        x_grad = w_grad = None
        if ctx.needs_input_grad[0]:
            x_grad = torch.nn.grad.conv2d_input(x.shape, w, grad_output, stride=stride.item(), padding=padding.item(), groups=groups.item(), dilation=dilation.item())
        if ctx.needs_input_grad[1]:
            w_grad = torch.nn.grad.conv2d_weight(x, w.shape, grad_output, stride=stride.item(), padding=padding.item(), groups=groups.item(), dilation=dilation.item())
            w_grad *= mask
        return x_grad, w_grad, None, None, None, None, None

class IrregConv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, mask_fn = None, weights_per_kernel = 4):
        super(IrregConv2D, self).__init__(in_channels, out_channels, kernel_size)
        self.mask = mask_fn(out_channels, in_channels, kernel_dims = kernel_size, weights_per_kernel = weights_per_kernel)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        weights = self.weight
        self.weight = nn.Parameter(weights*self.mask)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return BrainDamage.apply(input, self.weight, self.mask, self.padding, self.stride, self.groups, self.dilation)