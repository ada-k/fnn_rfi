"""Count MACs (multiply-accumulate ops) for all models via forward hooks."""
import sys
import os
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fnn_rfi.models.fnn_seg import FNNSeg, count_params
from fnn_rfi.models.fnn_inpaint import FNNInpaint
from fnn_rfi.models.fnn import FNN
from fnn_rfi.models.unet import UNet
from fnn_rfi.models.resnet_baseline import ResUNet, ResNetClassifier
from fnn_rfi.models.cnn import SimpleCNN
from fnn_rfi.models.rnn import LSTMClassifier
from fnn_rfi.models.transformer import TinyVisionTransformer


def count_macs(model, x):
    total = [0]
    hooks = []

    def hook_fn(module, inp, out):
        if isinstance(module, torch.nn.Conv2d):
            o = out.shape
            k = module.kernel_size
            total[0] += (o[2] * o[3] * k[0] * k[1]
                         * (module.in_channels // module.groups)
                         * module.out_channels)
        elif isinstance(module, torch.nn.Linear):
            total[0] += module.in_features * module.out_features
        elif isinstance(module, torch.nn.BatchNorm2d):
            total[0] += 2 * out.numel() // out.shape[0]

    for m in model.modules():
        hooks.append(m.register_forward_hook(hook_fn))
    model.eval()
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return total[0]


def main():
    seg_x = torch.randn(1, 1, 64, 64)
    cls_x = torch.randn(1, 3, 32, 32)

    print(f"{'Model':<32} {'Task':<14} {'Params':>10} {'MACs (M)':>10}")
    print("-" * 70)

    seg_models = [
        ("FNNSeg (fib, s=3)", FNNSeg(1, 16, 3, use_fib=True)),
        ("FNNSeg (std, s=3)", FNNSeg(1, 16, 3, use_fib=False)),
        ("FNNSeg (sep, s=3)", FNNSeg(1, 16, 3, use_fib=True, use_separable=True)),
        ("FNNSeg (fib, s=2)", FNNSeg(1, 16, 2, use_fib=True)),
        ("FNNSeg (fib, s=4)", FNNSeg(1, 16, 4, use_fib=True)),
        ("FNNSeg (fib, ch=8)", FNNSeg(1, 8, 3, use_fib=True)),
        ("FNNSeg (fib, ch=32)", FNNSeg(1, 32, 3, use_fib=True)),
        ("UNet (d=3, ch=8)", UNet(1, 8, 3)),
        ("UNet (d=4, ch=16)", UNet(1, 16, 4)),
        ("ResUNet (s=3, ch=16)", ResUNet(1, 16, 3)),
        ("FNNInpaint (fib, s=3)", FNNInpaint(1, 16, 3)),
    ]
    for name, m in seg_models:
        macs = count_macs(m, seg_x)
        p = count_params(m)
        print(f"{name:<32} {'seg':<14} {p:>10,} {macs/1e6:>10.2f}")

    print()
    cls_models = [
        ("FNN (fib, s=3)", FNN(3, 10, 3, 8, use_fib=True)),
        ("FNN (std, s=3)", FNN(3, 10, 3, 8, use_fib=False)),
        ("FNN (sep, s=3)", FNN(3, 10, 3, 8, use_fib=True, use_separable=True)),
        ("SimpleCNN", SimpleCNN(3, 10, 8)),
        ("LSTMClassifier", LSTMClassifier(3, 128, 1, 10)),
        ("TinyViT (d=2, e=64)", TinyVisionTransformer(3, 32, 4, 64, 2, 4, 10)),
        ("ResNetCls (ch=16, s=3)", ResNetClassifier(3, 10, 16, 3)),
    ]
    for name, m in cls_models:
        macs = count_macs(m, cls_x)
        p = count_params(m)
        print(f"{name:<32} {'cls':<14} {p:>10,} {macs/1e6:>10.2f}")


if __name__ == "__main__":
    main()
