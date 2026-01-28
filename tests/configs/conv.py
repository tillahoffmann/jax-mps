import numpy
from jax import lax

from .util import OperationTestConfig


def make_conv_op_configs():
    with OperationTestConfig.module_name("conv"):
        return [
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(1, 1),
                    padding="SAME",
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                ),
                numpy.random.normal(size=(2, 8, 8, 3)).astype(numpy.float32),
                numpy.random.normal(size=(3, 3, 3, 8)).astype(numpy.float32),
            ),
            OperationTestConfig(
                lambda lhs, rhs: lax.conv(lhs, rhs, (1, 1), "SAME"),
                numpy.random.normal(size=(1, 3, 8, 8)),
                numpy.random.normal(size=(16, 3, 3, 3)),
                name="lax.conv-SAME",
            ),
        ]
