from __future__ import annotations
import torch

def _validate_dtype(dtype: torch.dtype) -> None:
    """Autograd examples require a floating-point dtype."""
    if not dtype.is_floating_point:
        raise TypeError(
            f"Expected floating-point dtype, got {dtype}"
        )


def _to_float(tensor: torch.Tensor) -> float:
    """Convert a scalar Tensor to a Python float."""
    return tensor.detach().cpu().item()


def inspect_graph(
    device : str | torch.device = "cpu" , 
    dtype: torch.dtype = torch.float32
    ):

    x = torch.tensor(
        2.0,
        dtype = dtype,
        device = device,
        requires_grad = True
    )

    y = x * 3
    y.retain_grad()

    loss = y ** 2

    x_grad_before_backward = x.grad
    y_grad_before_backward = y.grad

    loss.backward()

    if x.grad is None:
        raise RuntimeError("Expected leaf tensor x to have a gradient")

    if y.grad is None:
        raise RuntimeError(
            "Expected y.grad after calling y.retain_grad()"
        )
    
    return {
        # 图节点属性
        "x_is_leaf": x.is_leaf,
        "x_requires_grad": x.requires_grad,
        "x_grad_fn": (
            None
            if x.grad_fn is None
            else type(x.grad_fn).__name__
        ),
        "y_is_leaf": y.is_leaf,
        "y_requires_grad": y.requires_grad,
        "y_grad_fn": (
            None
            if y.grad_fn is None
            else type(y.grad_fn).__name__
        ),
        "loss_is_leaf": loss.is_leaf,
        "loss_grad_fn": (
            None
            if loss.grad_fn is None
            else type(loss.grad_fn).__name__
        ),

        # backward 前后的梯度状态
        "x_grad_before_backward": x_grad_before_backward,
        "y_grad_before_backward": y_grad_before_backward,
        "x_grad": x.grad.detach().cpu().item(),
        "y_grad": y.grad.detach().cpu().item(),

        # 前向结果
        "x_value": x.detach().cpu().item(),
        "y_value": y.detach().cpu().item(),
        "loss_value": loss.detach().cpu().item(),

        # 设备与类型
        "device": str(x.device),
        "dtype": str(x.dtype),
    }

def gradient_accumulation_demo(
    device : str | torch.device = "cpu" , 
    dtype: torch.dtype = torch.float32
    ):

    w = torch.tensor (
        2.0,
        dtype = dtype,
        device = device,
        requires_grad = True
    )

    first_loss = w ** 2
    first_loss.backward()

    if w.grad is None:
        raise RuntimeError("Expected gradient after backward")

    first_grad = _to_float(w.grad)

    second_loss = w ** 2
    second_loss.backward()

    if w.grad is None:
        raise RuntimeError("Expected accumulated gradient")

    accumulated_grad = _to_float(w.grad)

    w.grad = None
    grad_is_none_after_clear = w.grad is None

    reset_loss = w**2
    reset_loss.backward()

    if w.grad is None:
        raise RuntimeError("Expected gradient after reset")

    reset_grad = _to_float(w.grad)

    return {
        "w_is_leaf": w.is_leaf,
        "w_grad_fn": (
            None
            if w.grad_fn is None
            else type(w.grad_fn).__name__
        ),
        "first_grad": first_grad,
        "accumulated_grad": accumulated_grad,
        "grad_is_none_after_clear": grad_is_none_after_clear,
        "reset_grad": reset_grad,
        "device": str(w.device),
        "dtype": str(w.dtype),
    }

def detach_demo(
        device : str | torch.device = "cpu" , 
        dtype: torch.dtype = torch.float32
    ):

    x = torch.arange(
        1,
        5,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    y = x * 2

    detached = y.detach()
    detached_cloned = y.detach().clone()

    return {
        "x_requires_grad": x.requires_grad,
        "x_is_leaf": x.is_leaf,

        "y_requires_grad": y.requires_grad,
        "y_is_leaf": y.is_leaf,
        "y_grad_fn": (
            None
            if y.grad_fn is None
            else type(y.grad_fn).__name__
        ),

        "detached_requires_grad": (
            detached.requires_grad
        ),
        "detached_is_leaf": detached.is_leaf,
        "detached_grad_fn": (
            None
            if detached.grad_fn is None
            else type(detached.grad_fn).__name__
        ),

        "cloned_requires_grad": (
            detached_cloned.requires_grad
        ),
        "cloned_is_leaf": detached_cloned.is_leaf,
        "cloned_grad_fn": (
            None
            if detached_cloned.grad_fn is None
            else type(detached_cloned.grad_fn).__name__
        ),

        # detach() 通常共享原 Tensor 的 Storage。
        "detach_shares_storage": (
            y.data_ptr() == detached.data_ptr()
        ),

        # clone() 创建独立 Storage。
        "clone_shares_storage": (
            y.data_ptr() == detached_cloned.data_ptr()
        ),

        "same_values_after_detach": torch.equal(
            y.detach(),
            detached,
        ),
        "same_values_after_clone": torch.equal(
            y.detach(),
            detached_cloned,
        ),

        "device": str(y.device),
        "dtype": str(y.dtype),
    }

def training_step_demo(
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 7,
    ):

    _validate_dtype(dtype)
    device = torch.device(device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
    
    torch.manual_seed(seed)

    model = torch.nn.Linear(
        in_features=3,
        out_features=1,
    ).to(
        device = device,
        dtype = dtype
    )

    x = torch.randn(
        32,
        3,
        device = device,
        dtype = dtype
    )

    true_weight = torch.tensor(
        [
            [5.0],
            [2.0],
            [1.0]
        ],
        device = device,
        dtype = dtype
    )

    true_bias = torch.tensor(
        [9.9],
        device = device,
        dtype = dtype
    )

    target = x @ true_weight + true_bias

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = 0.05
    )

    model.train()

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    optimizer.zero_grad(set_to_none=True)

    prediction = model(x)
    loss = loss_fn(prediction, target)

    initial_loss = _to_float(loss)
    
    loss.backward()

    parameters = list(model.parameters())

    all_grads_present = all(
        parameter.grad is not None
        for parameter in parameters
    )

    all_grads_finite = all(
        parameter.grad is not None 
        and torch.isfinite(parameter.grad).all().item()
        for parameter in parameters
    )

    grad_norm_squarred = torch.stack(
        [
            parameter.grad.detach()
            .float()
            .pow(2)
            .sum()
            for parameter in parameters
            if parameter.grad is not None
        ]
    ).sum()

    grad_norm = torch.sqrt(grad_norm_squarred)

    optimizer.step()

    parameters_deltas = torch.stack(
        [
            (parameter - before).abs().max().float()
            for parameter,before in zip(
                parameters,
                parameters_before,
            )
        ]
    )

    max_parameter_delta = parameters_deltas.max()

    with torch.no_grad():
        updated_prediction = model(x)
        updated_loss = loss_fn(updated_prediction, target)

    model_device = next(model.parameters()).device

    same_device = (
        model_device
        == x.device
        == target.device
        == prediction.device
    )

    return {
        "initial_loss": initial_loss,
        "final_loss": _to_float(updated_loss),

        "grad_norm": _to_float(grad_norm),
        "all_grads_present": all_grads_present,
        "all_grads_finite": all_grads_finite,

        "max_parameter_delta": _to_float(
            max_parameter_delta
        ),

        "model_device": str(model_device),
        "input_device": str(x.device),
        "target_device": str(target.device),
        "output_device": str(prediction.device),
        "same_device": same_device,

        "input_shape": tuple(x.shape),
        "target_shape": tuple(target.shape),
        "output_shape": tuple(prediction.shape),

        "dtype": str(x.dtype),
    }
    
def inference_mode_demo(
    device:str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 7
):
    _validate_dtype(dtype)
    device = torch.device(device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
    
    torch.manual_seed(seed)

    model = torch.nn.Sequential(
        torch.nn.Linear(4,8),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(8,2),
    ).to(
        dtype = dtype,
        device = device
    )

    x = torch.randn(
        32,
        4,
        device = device,
        dtype = dtype,
        requires_grad = True
    )

    model.eval()

    with torch.inference_mode():
        first_output = model(x)
        second_output = model(x)
    
    all_modules_in_eval = all(
        not module.training
        for module in model.modules()
    )

    return {
        "model_training": model.training,
        "all_modules_in_eval": all_modules_in_eval,

        "output_requires_grad": (
            first_output.requires_grad
        ),
        "output_grad_fn": (
            None
            if first_output.grad_fn is None
            else type(first_output.grad_fn).__name__
        ),

        # eval 后 Dropout 关闭，同一输入输出应该相同。
        "repeated_outputs_equal": torch.equal(
            first_output,
            second_output,
        ),

        "output_shape": tuple(first_output.shape),
        "device": str(first_output.device),
        "dtype": str(first_output.dtype),
    }

def main() -> None:
    result = inference_mode_demo()

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
