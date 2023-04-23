"""
From https://github.com/valeoai/MVRSS
"""
from torch import nn
import torch 
import torch.nn.functional as F

from typing import Optional

class SmoothCELoss(nn.Module):
    """
    Smooth cross entropy loss
    SCE = SmoothL1Loss() + BCELoss()
    By default reduction is mean. 
    """
    def __init__(self, alpha):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.bce = nn.BCELoss()
        self.alpha = alpha
    
    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + \
                (1-self.alpha) * self.smooth_l1(input, target)


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros(shape[0], num_classes, *shape[1:],
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py

def soft_dice_loss(input: torch.Tensor, target: torch.Tensor, eps: float = 1e-8,
                   global_weight: float = 1.) -> torch.Tensor:
    r"""Function that computes Sørensen-Dice Coefficient loss.
    Arthur: add ^2 for soft formulation

    See :class:`~kornia.losses.DiceLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                         .format(input.shape))

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError("input and target shapes must be the same. Got: {} and {}"
                         .format(input.shape, input.shape))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    cardinality = torch.sum(torch.pow(input_soft, 2) + torch.pow(target_one_hot, 2), dims)

    dice_score = 2. * intersection / (cardinality + eps)
    return global_weight*torch.mean(-dice_score + 1.)


class SoftDiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.
    Arthur: add ^2 for soft formulation

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, global_weight: float = 1.) -> None:
        super(SoftDiceLoss, self).__init__()
        self.eps: float = 1e-6
        self.global_weight = global_weight

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        return soft_dice_loss(input, target, self.eps, self.global_weight)

class CoherenceLoss(nn.Module):
    """
    Compute the Unsupervised Coherence Loss

    PARAMETERS
    ----------
    global_weight: float
        Global weight to apply on the entire loss
    """

    def __init__(self, global_weight: float = 1.) -> None:
        super(CoherenceLoss, self).__init__()
        self.global_weight = global_weight
        self.mse = nn.MSELoss()

    def forward(self, rd_input: torch.Tensor,
                ra_input: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute the loss between the two predicted view masks"""
        rd_softmax = nn.Softmax(dim=1)(rd_input)
        ra_softmax = nn.Softmax(dim=1)(ra_input)
        rd_range_probs = torch.max(rd_softmax, dim=3, keepdim=True)[0]
        # Rotate RD Range vect to match zero range
        rd_range_probs = torch.rot90(rd_range_probs, 2, [2, 3])
        ra_range_probs = torch.max(ra_softmax, dim=3, keepdim=True)[0]
        coherence_loss = self.mse(rd_range_probs, ra_range_probs)
        weighted_coherence_loss = self.global_weight*coherence_loss
        return weighted_coherence_loss
