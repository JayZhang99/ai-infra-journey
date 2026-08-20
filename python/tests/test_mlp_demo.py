from pathlib import Path
import pytest
import torch

from python.mlp_demo import (
    TinyMLP,
    make_data,
    train_model,
    save_and_reload,
)

def test_output_shape():
    torch.manual_seed(7)
    x , _ = make_data()
    model = TinyMLP()
    output = model.forward(x)
    
    assert tuple(x.shape) == (512, 8)
    assert tuple(output.shape) == (512, 2)
    
def test_parameters_are_registerd():
    model = TinyMLP()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 354

def test_loss_decreases():
    torch.manual_seed(7)
    x, target = make_data()
    model = TinyMLP()
    losses = train_model(model, x, target)
    assert len(losses) == 200
    assert torch.isfinite(torch.tensor(losses)).all()
    assert losses[-1] < losses[0] * 0.1

def test_state_dict_roundrip(tmp_path):
    torch.manual_seed(7)
    x, target = make_data()
    model = TinyMLP()
    losses = train_model(model, x, target)
    loaded = save_and_reload(model, tmp_path/"model.pt")
    model.eval()
    loaded.eval()
    with torch.inference_mode():
        expected = model(x)
        actual = loaded(x)
    max_error = (expected - actual).abs().max().item()
    assert max_error < 1e-6

def test_eval_mode_propagates():
    torch.manual_seed(7)
    x, target = make_data()
    model = TinyMLP()
    losses = train_model(model, x, target)
    model.eval()
    assert model.training is False
    assert all(not module.training for module in model.modules())
    

def test_inference_output_has_no_grad():
    torch.manual_seed(7)
    x, target = make_data()
    model = TinyMLP()
    losses = train_model(model, x, target)
    model.eval()
    with torch.inference_mode():
        prediction = model(x)
    
    assert prediction.requires_grad is False
    assert prediction.grad_fn is None

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)

def test_cuda_smoke():
    device = torch.device("cuda:0")
    torch.manual_seed(7)
    x, target = make_data()
    model = TinyMLP()
    cuda_x = x.to(device)
    cuda_target = target.to(device)
    cuda_model = model.to(device)
    losses = train_model(model, x, target)
    cuda_losses = train_model(cuda_model, cuda_x, cuda_target)
    cpu_cuda_losses = cuda_losses.cpu()
    assert cuda_losses[-1] < losses[-1]