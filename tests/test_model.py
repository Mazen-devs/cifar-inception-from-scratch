"""Sanity tests for the architecture. Run with: pytest -q"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import CIFARInception, InceptionModule, count_parameters  # noqa: E402


def test_parameter_count_matches_notebook():
    # The notebook prints "Total trainable parameters: 549,802"
    assert count_parameters(CIFARInception()) == 549_802


def test_output_shape_is_batch_by_classes():
    model = CIFARInception().eval()
    with torch.no_grad():
        logits = model(torch.randn(4, 3, 32, 32))
    assert logits.shape == (4, 10)


def test_shape_trace_matches_design():
    model = CIFARInception().eval()
    with torch.no_grad():
        _, trace = model(torch.randn(1, 3, 32, 32), return_trace=True)
    assert tuple(trace["stem (after stride-2 pool)"]) == (1, 64, 16, 16)
    assert tuple(trace["inception 2b"]) == (1, 256, 8, 8)
    assert tuple(trace["inception 3a"]) == (1, 304, 4, 4)


def test_inception_module_concatenates_all_branches():
    module = InceptionModule(64, 24, 32, 48, 8, 16, 24)
    out = module(torch.randn(2, 64, 16, 16))
    assert out.shape[1] == module.out_channels == 112


def test_gradients_reach_the_stem():
    model = CIFARInception()
    loss = model(torch.randn(2, 3, 32, 32)).sum()
    loss.backward()
    assert model.stem[0].conv.weight.grad is not None
    assert torch.isfinite(model.stem[0].conv.weight.grad).all()
