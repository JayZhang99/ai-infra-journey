import math
import torch
import pytest

from python.autograd_playground import (
    inspect_graph,
    gradient_accumulation_demo,
    detach_demo,
    training_step_demo,
    inference_mode_demo,
)

def test_inspect_graph():
    result = inspect_graph()
    assert result["x_is_leaf"] is True
    assert result["x_grad_fn"] == None
    assert result["y_is_leaf"] is False
    assert result["y_grad_fn"] != None
    assert result["x_grad"] == pytest.approx(36.0)
    assert result["y_grad"] == pytest.approx(12.0)

def test_gradient_accumulation_demo():
    result = gradient_accumulation_demo()
    assert result["first_grad"] == pytest.approx(4.0)
    assert result["accumulated_grad"] == pytest.approx(8.0)
    assert result["reset_grad"] == pytest.approx(4.0)
    assert result["grad_is_none_after_clear"] is True

def test_detach_demo():
    result = detach_demo()
    assert result["detached_requires_grad"] is False
    assert result["detach_shares_storage"] is True
    assert result["clone_shares_storage"] is False

def test_trainning_step_demo():
    result = training_step_demo()
    assert math.isfinite(result["grad_norm"])
    assert result["max_parameter_delta"] > 0

def test_inference_mode_demo():
    result = inference_mode_demo()
    assert result["output_requires_grad"] is False
    assert result["output_grad_fn"] == None

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)

def test_all_demos_on_cuda():
    graph_result = inspect_graph("cuda:0")
    accumulation_result = (
        gradient_accumulation_demo("cuda:0")
    )
    detach_result = detach_demo("cuda:0")
    training_result = training_step_demo("cuda:0")
    inference_result = inference_mode_demo("cuda:0")

    assert graph_result["device"] == "cuda:0"
    assert graph_result["x_grad"] == pytest.approx(36.0)
    assert graph_result["y_grad"] == pytest.approx(12.0)

    assert (
        accumulation_result["accumulated_grad"]
        == pytest.approx(8.0)
    )

    assert detach_result["detach_shares_storage"] is True
    assert detach_result["clone_shares_storage"] is False

    assert training_result["same_device"] is True
    assert training_result["model_device"] == "cuda:0"
    assert training_result["max_parameter_delta"] > 0

    assert inference_result["device"] == "cuda:0"
    assert inference_result["output_requires_grad"] is False