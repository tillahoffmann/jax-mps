import pytest
from jax import numpy as jnp
from jax import random
from numpyro import distributions as dists

from .util import OperationTestConfig, xfail_match


class NumpyroDistributionTestConfig(OperationTestConfig):
    """
    Test config for numpyro distributions that evaluates log_prob and samples.

    Args:
        dist_cls: Distribution class to test.
        *args: Positional arguments passed to the distribution constructor.
        **kwargs: Keyword arguments passed to OperationTestConfig.
    """

    def __init__(self, dist_cls: type[dists.Distribution], *args, **kwargs) -> None:
        kwargs.setdefault("name", dist_cls.__name__)
        super().__init__(
            lambda x, *args: dist_cls(*args).log_prob(x).mean(),
            lambda key: dist_cls(*[a(key) if callable(a) else a for a in args]).sample(
                random.key(17)
            ),  # pyright: ignore[reportArgumentType]
            *args,
            **kwargs,
        )


def make_numpyro_op_configs():
    with OperationTestConfig.module_name("numpyro"):
        for batch_shape in [(), (3,)]:
            yield from [
                NumpyroDistributionTestConfig(
                    dists.Normal,
                    lambda key, bs=batch_shape: random.normal(key, bs),
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Gamma,
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Exponential,
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Uniform,
                    lambda key, bs=batch_shape: random.normal(key, bs),
                    lambda key, bs=batch_shape: random.normal(key, bs) + 2,
                ),
                NumpyroDistributionTestConfig(
                    dists.Laplace,
                    lambda key, bs=batch_shape: random.normal(key, bs),
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Cauchy,
                    lambda key, bs=batch_shape: random.normal(key, bs),
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.HalfNormal,
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.HalfCauchy,
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.LogNormal,
                    lambda key, bs=batch_shape: random.normal(key, bs),
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Beta,
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.StudentT,
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs)
                    + 2,  # df > 2
                    lambda key, bs=batch_shape: random.normal(key, bs),
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Dirichlet,
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs + (3,)),
                ),
                # Discrete distributions (differentiable_argnums excludes sample at 0).
                NumpyroDistributionTestConfig(
                    dists.BernoulliProbs,
                    lambda key, bs=batch_shape: random.uniform(
                        key, bs, minval=0.1, maxval=0.9
                    ),
                    differentiable_argnums=(1,),
                    name="Bernoulli",
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.BinomialProbs,
                        lambda key, bs=batch_shape: random.uniform(
                            key, bs, minval=0.1, maxval=0.9
                        ),
                        10,  # total_count (not differentiable)
                        differentiable_argnums=(1,),
                        name="Binomial",
                    ),
                    marks=[xfail_match("gather:.+unsupported gather pattern")],
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.CategoricalProbs,
                        lambda key, bs=batch_shape: random.dirichlet(
                            key, jnp.ones(5), bs
                        ),
                        differentiable_argnums=(1,),
                        name="Categorical",
                    ),
                    marks=[
                        xfail_match("Unsupported operation.+stablehlo.reduce_window")
                    ],
                ),
                NumpyroDistributionTestConfig(
                    dists.Poisson,
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),
                    differentiable_argnums=(1,),
                ),
                NumpyroDistributionTestConfig(
                    dists.GeometricProbs,
                    lambda key, bs=batch_shape: random.uniform(
                        key, bs, minval=0.1, maxval=0.9
                    ),
                    differentiable_argnums=(1,),
                    name="Geometric",
                ),
                NumpyroDistributionTestConfig(
                    dists.NegativeBinomial2,
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),  # mean
                    lambda key, bs=batch_shape: random.gamma(
                        key, 5.0, bs
                    ),  # concentration
                    differentiable_argnums=(1, 2),
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.MultinomialProbs,
                        lambda key, bs=batch_shape: random.dirichlet(
                            key, jnp.ones(5), bs
                        ),
                        10,  # total_count (not differentiable)
                        differentiable_argnums=(1,),
                        name="Multinomial",
                    ),
                    marks=[
                        xfail_match("Unsupported operation.+stablehlo.reduce_window")
                    ],
                ),
                # Additional continuous distributions.
                NumpyroDistributionTestConfig(
                    dists.Gumbel,
                    lambda key, bs=batch_shape: random.normal(key, bs),
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Logistic,
                    lambda key, bs=batch_shape: random.normal(key, bs),
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Pareto,
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),  # scale
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),  # alpha
                ),
                NumpyroDistributionTestConfig(
                    dists.Weibull,
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),  # scale
                    lambda key, bs=batch_shape: random.gamma(
                        key, 5.0, bs
                    ),  # concentration
                ),
                NumpyroDistributionTestConfig(
                    dists.Chi2,
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs) + 2,  # df
                ),
                NumpyroDistributionTestConfig(
                    dists.InverseGamma,
                    lambda key, bs=batch_shape: random.gamma(
                        key, 5.0, bs
                    ),  # concentration
                    lambda key, bs=batch_shape: random.gamma(key, 5.0, bs),  # rate
                ),
                NumpyroDistributionTestConfig(
                    dists.VonMises,
                    lambda key, bs=batch_shape: random.uniform(
                        key, bs, minval=-jnp.pi, maxval=jnp.pi
                    ),  # loc
                    lambda key, bs=batch_shape: random.gamma(
                        key, 5.0, bs
                    ),  # concentration
                ),
                # Multivariate distributions.
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.MultivariateNormal,
                        lambda key, bs=batch_shape: random.normal(
                            key, bs + (4,)
                        ),  # loc
                        None,  # covariance_matrix
                        None,  # precision_matrix
                        lambda key: jnp.linalg.cholesky(jnp.eye(4) + jnp.ones((4, 4))),
                    ),
                    marks=[
                        xfail_match(
                            "gather:.+unsupported gather pattern|Native op handler returned nil"
                        )
                    ],
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.LowRankMultivariateNormal,
                        lambda key, bs=batch_shape: random.normal(
                            key, bs + (4,)
                        ),  # loc
                        lambda key, bs=batch_shape: random.normal(
                            key, bs + (4, 2)
                        ),  # cov_factor
                        lambda key, bs=batch_shape: random.gamma(
                            key, 5.0, bs + (4,)
                        ),  # cov_diag
                    ),
                    marks=[xfail_match("scatter:.+unsupported scatter pattern")],
                ),
            ]
