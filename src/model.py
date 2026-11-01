"""CIFARInception: an Inception-style CNN designed for native 32x32 CIFAR-10 images.

Architecture code is taken verbatim from the training notebook
(notebooks/cifar_inception.ipynb, Steps 6-8). Total trainable parameters: 549,802.
"""
import torch
import torch.nn as nn


class ConvBnRelu(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU. Bias omitted (redundant given BatchNorm)."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class InceptionModule(nn.Module):
    """Four parallel branches (1x1 / 3x3-reduced / 5x5-reduced / pooled-projection),
    concatenated along the channel dimension."""

    def __init__(self, in_ch, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj):
        super().__init__()
        self.branch1 = ConvBnRelu(in_ch, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBnRelu(in_ch, red_3x3, kernel_size=1),
            ConvBnRelu(red_3x3, out_3x3, kernel_size=3, padding=1))
        self.branch3 = nn.Sequential(
            ConvBnRelu(in_ch, red_5x5, kernel_size=1),
            ConvBnRelu(red_5x5, out_5x5, kernel_size=5, padding=2))
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_ch, pool_proj, kernel_size=1))
        self.out_channels = out_1x1 + out_3x3 + out_5x5 + pool_proj

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], dim=1)


class CIFARInception(nn.Module):
    def __init__(self, num_classes=10, dropout_p=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBnRelu(3, 32, kernel_size=3, padding=1),
            ConvBnRelu(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2))                                             # 32->16

        self.inc1a = InceptionModule(64,  24, 32, 48,  8, 16, 24)           # -> 112
        self.inc1b = InceptionModule(112, 32, 48, 64,  8, 16, 16)           # -> 128
        self.pool1 = nn.MaxPool2d(2, 2)                                     # 16->8

        self.inc2a = InceptionModule(128, 64, 64, 96, 16, 32, 32)           # -> 224
        self.inc2b = InceptionModule(224, 64, 96, 128, 16, 32, 32)          # -> 256
        self.pool2 = nn.MaxPool2d(2, 2)                                     # 8->4

        self.inc3a = InceptionModule(256, 96, 96, 144, 16, 32, 32)          # -> 304

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(304, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_trace=False):
        trace = {}
        x = self.stem(x);   trace["stem (after stride-2 pool)"] = x.shape
        x = self.inc1a(x);  trace["inception 1a"] = x.shape
        x = self.inc1b(x);  trace["inception 1b"] = x.shape
        x = self.pool1(x);  trace["pool 1"] = x.shape
        x = self.inc2a(x);  trace["inception 2a"] = x.shape
        x = self.inc2b(x);  trace["inception 2b"] = x.shape
        x = self.pool2(x);  trace["pool 2"] = x.shape
        x = self.inc3a(x);  trace["inception 3a"] = x.shape
        x = self.gap(x);    trace["global avg pool"] = x.shape
        x = torch.flatten(x, 1); trace["flatten"] = x.shape
        x = self.dropout(x)
        logits = self.classifier(x); trace["logits"] = logits.shape
        return (logits, trace) if return_trace else logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = CIFARInception()
    print(f"Trainable parameters: {count_parameters(m):,}")
    _, trace = m(torch.randn(1, 3, 32, 32), return_trace=True)
    for k, v in trace.items():
        print(f"{k:28s} -> {tuple(v)}")
