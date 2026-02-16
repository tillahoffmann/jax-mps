import jax
import numpy
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from .configs import OperationTestConfig, make_benchmark_op_configs

pytestmark = pytest.mark.benchmark

OPERATION_TEST_CONFIGS = list(make_benchmark_op_configs())
GRAD_TEST_CONFIGS = []
for op_config in OPERATION_TEST_CONFIGS:
    with jax.default_device("cpu"):
        differentiable_argnums = op_config.get_differentiable_argnums()
        GRAD_TEST_CONFIGS.extend(
            (op_config, argnum) for argnum in differentiable_argnums
        )

DEVICES = []
for platform in ["cpu", "mps"]:
    try:
        DEVICES.append(jax.devices(platform)[0])
    except RuntimeError as ex:
        if "Unknown backend" not in str(ex):
            raise


@pytest.fixture(params=DEVICES, ids=lambda x: x.platform)
def device(request: pytest.FixtureRequest):
    return request.param


@pytest.mark.parametrize(
    "op_config", OPERATION_TEST_CONFIGS, ids=lambda op_config: op_config.name
)
def test_benchmark_value(
    op_config: OperationTestConfig, device, benchmark: BenchmarkFixture
) -> None:
    # Get the args and move them to the right device.
    rng = numpy.random.default_rng(op_config.seed)
    args = jax.tree.map(
        lambda x: jax.device_put(x, device).block_until_ready(), op_config.get_args(rng)
    )
    kwargs = jax.tree.map(
        lambda x: jax.device_put(x, device).block_until_ready(),
        op_config.get_kwargs(rng),
    )
    func = jax.jit(op_config.func, static_argnums=op_config.static_argnums)

    def run():
        return func(*args, **kwargs).block_until_ready()

    # Run once for jit-compile and once for safety.
    run()
    run()

    # Then benchmark.
    benchmark(run)


@pytest.mark.parametrize(
    "op_config,argnum",
    GRAD_TEST_CONFIGS,
    ids=[f"{cfg.name}_grad{argnum}" for cfg, argnum in GRAD_TEST_CONFIGS],
)
def test_benchmark_grad(
    op_config: OperationTestConfig, argnum: int, device, benchmark: BenchmarkFixture
) -> None:
    # Get the args and move them to the right device.
    rng = numpy.random.default_rng(op_config.seed)
    args = jax.tree.map(
        lambda x: jax.device_put(x, device).block_until_ready(), op_config.get_args(rng)
    )
    kwargs = jax.tree.map(
        lambda x: jax.device_put(x, device).block_until_ready(),
        op_config.get_kwargs(rng),
    )

    # Build scalar loss function (grad requires scalar output).
    func = op_config.func

    def scalar_output(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, (tuple, list)):
            result = result[0]
        # Handle complex outputs (same as OperationTestConfig.evaluate_grad).
        if result.dtype == jax.numpy.complex64:
            result = jax.numpy.abs(result)
        if result.shape != ():
            result = result.mean()
        return result

    grad_func = jax.jit(
        op_config.grad_transform(scalar_output, argnums=argnum),
        static_argnums=op_config.static_argnums,
    )

    def run():
        result = grad_func(*args, **kwargs)
        jax.tree.map(lambda x: x.block_until_ready(), result)
        return result

    # Run once for jit-compile and once for safety.
    run()
    run()

    # Then benchmark.
    benchmark(run)
