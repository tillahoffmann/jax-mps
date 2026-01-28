import numpy
from flax import nnx

from .util import OperationTestConfig


def _call_module(module, *args, **kwargs):
    """Forward-pass a module. The module is an argument (not the op) so that
    grad transforms like ``nnx.grad`` can manage its graph state."""
    return module(*args, **kwargs)


def make_flax_op_configs():
    _GT = nnx.grad  # grad transform that handles nnx graph nodes

    with OperationTestConfig.module_name("flax"):
        return [
            OperationTestConfig(
                _call_module,
                lambda: nnx.Linear(3, 4, rngs=nnx.Rngs(17)),
                numpy.random.standard_normal((10, 3)),
                name="nnx.Linear",
                grad_transform=_GT,
            ),
            OperationTestConfig(
                _call_module,
                lambda: nnx.Conv(3, 8, (3, 3), rngs=nnx.Rngs(17)),
                numpy.random.standard_normal((4, 28, 28, 3)),
                name="nnx.Conv",
                grad_transform=_GT,
            ),
            OperationTestConfig(
                _call_module,
                lambda: nnx.Conv(5, 16, (3, 3), (2, 2), rngs=nnx.Rngs(17)),
                numpy.random.standard_normal((2, 32, 32, 5)),
                name="nnx.Conv(strided)",
                grad_transform=_GT,
            ),
            OperationTestConfig(
                _call_module,
                lambda: nnx.Conv(5, 16, (5, 5), padding="VALID", rngs=nnx.Rngs(17)),
                numpy.random.standard_normal((2, 32, 32, 5)),
                name="nnx.Conv(valid-padding)",
                grad_transform=_GT,
            ),
            OperationTestConfig(
                _call_module,
                lambda: nnx.Conv(
                    3, 8, (3, 3), kernel_dilation=(2, 2), rngs=nnx.Rngs(19)
                ),
                numpy.random.standard_normal((2, 32, 32, 3)),
                name="nnx.Conv(dilated)",
                grad_transform=_GT,
            ),
            # 1x1 convolution (pointwise)
            OperationTestConfig(
                _call_module,
                lambda: nnx.Conv(64, 128, (1, 1), rngs=nnx.Rngs(17)),
                numpy.random.standard_normal((2, 16, 16, 64)),
                name="nnx.Conv(1x1)",
                grad_transform=_GT,
            ),
            # Depthwise convolution (feature_group_count = in_features)
            OperationTestConfig(
                _call_module,
                lambda: nnx.Conv(
                    16, 16, (3, 3), feature_group_count=16, rngs=nnx.Rngs(17)
                ),
                numpy.random.standard_normal((2, 28, 28, 16)),
                name="nnx.Conv(depthwise)",
                grad_transform=_GT,
            ),
            # Grouped convolution
            OperationTestConfig(
                _call_module,
                lambda: nnx.Conv(
                    16, 32, (3, 3), feature_group_count=4, rngs=nnx.Rngs(17)
                ),
                numpy.random.standard_normal((2, 28, 28, 16)),
                name="nnx.Conv(grouped)",
                grad_transform=_GT,
            ),
            # Strided + dilated + valid padding combined
            OperationTestConfig(
                _call_module,
                lambda: nnx.Conv(
                    8,
                    16,
                    (3, 3),
                    (2, 2),
                    kernel_dilation=(2, 2),
                    padding="VALID",
                    rngs=nnx.Rngs(17),
                ),
                numpy.random.standard_normal((2, 32, 32, 8)),
                name="nnx.Conv(strided+dilated+valid)",
                grad_transform=_GT,
            ),
            OperationTestConfig(
                _call_module,
                lambda: nnx.Embed(num_embeddings=100, features=5, rngs=nnx.Rngs(17)),
                numpy.random.randint(0, 100, (3, 4)),
                name="nnx.Embed",
                grad_transform=_GT,
            ),
            # BatchNorm
            OperationTestConfig(
                _call_module,
                lambda: nnx.BatchNorm(
                    num_features=16, momentum=0.9, epsilon=1e-5, rngs=nnx.Rngs(17)
                ),
                numpy.random.standard_normal((4, 16)),
                name="nnx.BatchNorm",
                grad_transform=_GT,
            ),
            # BatchNorm with spatial dimensions (like in CNN)
            OperationTestConfig(
                _call_module,
                lambda: nnx.BatchNorm(
                    num_features=8, momentum=0.9, epsilon=1e-5, rngs=nnx.Rngs(17)
                ),
                numpy.random.standard_normal((2, 28, 28, 8)),
                name="nnx.BatchNorm(spatial)",
                grad_transform=_GT,
            ),
        ]
