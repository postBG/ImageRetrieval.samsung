from copy import deepcopy

import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage


class GreyToColor(object):
    """Convert Grey Image label to binary
    """

    def __call__(self, image):
        if len(image.size()) == 3 and image.size(0) == 1:
            return image.repeat([3, 1, 1])
        elif len(image.size()) == 2:
            image.unsqueeze_(0)
            return image.repeat([3, 1, 1])
        else:
            return image

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class ImageHelper(object):
    def __init__(self, mean, std):
        self.denormalize = Denormalize(mean=mean, std=std)
        self.to_pil_image = ToPILImage()
        self.grey_to_color = GreyToColor()

    def to_pil_image(self, t):
        t = deepcopy(t.detach())
        t = self.denormalize(t)
        t = self.grey_to_color(t)
        return self.to_pil_image(t)

    def show_img(self, t):
        pil_image = self.to_pil_image(t)
        plt.imshow(pil_image)
