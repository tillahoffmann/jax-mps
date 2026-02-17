from jax import lax, random

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
                lambda key: random.normal(key, (2, 8, 8, 3)),
                lambda key: random.normal(key, (3, 3, 3, 8)),
            ),
            OperationTestConfig(
                lambda lhs, rhs: lax.conv(lhs, rhs, (1, 1), "SAME"),
                lambda key: random.normal(key, (1, 3, 8, 8)),
                lambda key: random.normal(key, (16, 3, 3, 3)),
                name="lax.conv-SAME",
            ),
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(2,),
                    padding="SAME",
                    dimension_numbers=("NWC", "WIO", "NWC"),
                ),
                lambda key: random.normal(key, (2, 16, 3)),
                lambda key: random.normal(key, (3, 3, 4)),
                name="lax.conv_general_dilated-1d-NWC",
            ),
            # Strided 2D convolutions with SAME padding
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(2, 2),
                    padding="SAME",
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                ),
                lambda key: random.normal(key, (2, 8, 8, 3)),
                lambda key: random.normal(key, (3, 3, 3, 8)),
                name="lax.conv_general_dilated-stride2-SAME",
            ),
            # Asymmetric padding test
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(2, 2),
                    padding=((1, 0), (1, 0)),
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                ),
                lambda key: random.normal(key, (2, 8, 8, 3)),
                lambda key: random.normal(key, (3, 3, 3, 8)),
                name="lax.conv_general_dilated-stride2-asymmetric",
            ),
            # Larger kernel with stride
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(2, 2),
                    padding="SAME",
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                ),
                lambda key: random.normal(key, (2, 16, 16, 3)),
                lambda key: random.normal(key, (5, 5, 3, 8)),
                name="lax.conv_general_dilated-5x5-stride2",
            ),
            # Stride + dilation combined
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(2, 2),
                    padding="SAME",
                    rhs_dilation=(2, 2),
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                ),
                lambda key: random.normal(key, (2, 16, 16, 3)),
                lambda key: random.normal(key, (3, 3, 3, 8)),
                name="lax.conv_general_dilated-stride2-dilated",
            ),
            # VALID padding with stride
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(2, 2),
                    padding="VALID",
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                ),
                lambda key: random.normal(key, (2, 16, 16, 3)),
                lambda key: random.normal(key, (3, 3, 3, 8)),
                name="lax.conv_general_dilated-stride2-VALID",
            ),
            # 1D strided + dilated
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(2,),
                    padding="SAME",
                    rhs_dilation=(2,),
                    dimension_numbers=("NWC", "WIO", "NWC"),
                ),
                lambda key: random.normal(key, (2, 32, 3)),
                lambda key: random.normal(key, (3, 3, 4)),
                name="lax.conv_general_dilated-1d-stride2-dilated",
            ),
        ]
