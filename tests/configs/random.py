from jax import random

from .util import OperationTestConfig


def make_random_op_configs():
    with OperationTestConfig.module_name("random"):
        for shape in [(), (3,), (7, 8)]:
            yield from [
                OperationTestConfig(
                    random.normal, random.key(17), shape, static_argnums=(1,)
                ),
                OperationTestConfig(
                    random.truncated_normal,
                    random.key(17),
                    -0.1,
                    0.2,
                    shape,
                    static_argnums=(3,),
                ),
                OperationTestConfig(
                    random.uniform,
                    random.key(17),
                    (3,),
                    None,
                    0.3,
                    0.5,
                    static_argnums=(1,),
                ),
                OperationTestConfig(random.split, random.key(18)),
                OperationTestConfig(
                    random.split, random.key(18), 5, static_argnums=(1,)
                ),
            ]
