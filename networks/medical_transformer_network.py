from torch import nn

from networks.medical_transformer.axialnet import MedT, axialunet, gated

__all__ = ["MedicalAxialUNet", "MedicalGatedAxialUNet", "MedicalTransformer"]


class _MedicalTransformerWrapper(nn.Module):
    def __init__(self, factory, num_classes=1, in_channels=3, img_size=352):
        super().__init__()
        self.img_size = img_size
        self.model = factory(num_classes=num_classes, imgchan=in_channels, img_size=img_size)

    def forward(self, x):
        return self.model(x)


class MedicalAxialUNet(_MedicalTransformerWrapper):
    def __init__(self, num_classes=1, in_channels=3, img_size=352):
        super().__init__(axialunet, num_classes=num_classes, in_channels=in_channels, img_size=img_size)


class MedicalGatedAxialUNet(_MedicalTransformerWrapper):
    def __init__(self, num_classes=1, in_channels=3, img_size=352):
        super().__init__(gated, num_classes=num_classes, in_channels=in_channels, img_size=img_size)


class MedicalTransformer(_MedicalTransformerWrapper):
    def __init__(self, num_classes=1, in_channels=3, img_size=352):
        super().__init__(MedT, num_classes=num_classes, in_channels=in_channels, img_size=img_size)
