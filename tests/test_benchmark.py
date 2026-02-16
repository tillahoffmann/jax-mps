import jax
import numpy
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from .configs import OperationTestConfig, make_benchmark_op_configs

pytestmark = pytest.mark.benchmark

OPERATION_TEST_CONFIGS = list(make_benchmark_op_configs())
DEVICES = []
for platform in ["cpu", "mps"]:
    try:
        DEVICES.append(jax.devices(platform)[0])
    except RuntimeError as ex:
        if "Unknown backend" not in str(ex):
            raise


@pytest.fixture(params=OPERATION_TEST_CONFIGS, ids=lambda op_config: op_config.name)
def op_config(request: pytest.FixtureRequest) -> OperationTestConfig:
    return request.param


@pytest.fixture(params=DEVICES, ids=lambda x: x.platform)
def device(request: pytest.FixtureRequest):
    return request.param


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
