import random
import torchvision.transforms as T
import torchvision.transforms.functional as F


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.hflip(img)
            target = F.hflip(target)
        return img, target


class RandomResizedCrop(object):
    def __init__(self, size, scale, ratio=(3. / 4., 4. / 3.)):
        self.rrc_transform = T.RandomResizedCrop(size=size, scale=scale, ratio=ratio)

    def __call__(self, img, target=None):
        y1, x1, h, w = self.rrc_transform.get_params(img, self.rrc_transform.scale, self.rrc_transform.ratio)
        img = F.resized_crop(img, y1, x1, h, w, self.rrc_transform.size, F.InterpolationMode.BILINEAR)
        target = F.resized_crop(target, y1, x1, h, w, self.rrc_transform.size, F.InterpolationMode.NEAREST)
        return img, target


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), F.to_tensor(target)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
