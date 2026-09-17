import torch

_REGISTERED = False

def register_runtime_kernels():
    global _REGISTERED
    if _REGISTERED:
        return

    register_scale_add()
    register_reduce_sum()
    _REGISTERED = True

def register_scale_add():
    @torch.library.register_fake(
        "jay_ops::scale_add",
        allow_override=False
    )
    def fake(x, alpha):
        return torch.empty_like(x)
    
    def setup(ctx, inputs, output):
        x, alpha = inputs
        ctx.alpha = alpha
    
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None

    torch.library.register_autograd(
        "jay_ops::scale_add",
        backward, setup_context=setup
    )


def register_reduce_sum():
    @torch.library.register_fake(
        "jay_ops::reduce_sum",
        allow_override=False
    )

    def fake(x):
        return x.new_empty(())

    def setup(ctx, inputs, output):
        (x,) = inputs
        ctx.input_shape = x.shape

    def backward(ctx, grad_output):
        return grad_output.expand(ctx.input_shape)
    
    torch.library.register_autograd(
        "jay_ops::reduce_sum",
        backward, setup_context=setup
    )

  