import numpy
import pytest
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
            lambda rng: dist_cls(*[a(rng) if callable(a) else a for a in args]).sample(
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
                    lambda rng, bs=batch_shape: rng.standard_normal(bs),
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Gamma,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Exponential,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Uniform,
                    lambda rng, bs=batch_shape: rng.standard_normal(bs),
                    lambda rng, bs=batch_shape: rng.standard_normal(bs) + 2,
                ),
                NumpyroDistributionTestConfig(
                    dists.Laplace,
                    lambda rng, bs=batch_shape: rng.standard_normal(bs),
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Cauchy,
                    lambda rng, bs=batch_shape: rng.standard_normal(bs),
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.HalfNormal,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.HalfCauchy,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.LogNormal,
                    lambda rng, bs=batch_shape: rng.standard_normal(bs),
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Beta,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.StudentT,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs) + 2,  # df > 2
                    lambda rng, bs=batch_shape: rng.standard_normal(bs),
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Dirichlet,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs + (3,)),
                ),
                # Discrete distributions (differentiable_argnums excludes sample at 0).
                NumpyroDistributionTestConfig(
                    dists.BernoulliProbs,
                    lambda rng, bs=batch_shape: rng.uniform(0.1, 0.9, bs),
                    differentiable_argnums=(1,),
                    name="Bernoulli",
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.BinomialProbs,
                        lambda rng, bs=batch_shape: rng.uniform(0.1, 0.9, bs),
                        10,  # total_count (not differentiable)
                        differentiable_argnums=(1,),
                        name="Binomial",
                    ),
                    marks=[xfail_match("gather:.+unsupported gather pattern")],
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.CategoricalProbs,
                        lambda rng, bs=batch_shape: rng.dirichlet(numpy.ones(5), bs),
                        differentiable_argnums=(1,),
                        name="Categorical",
                    ),
                    marks=[
                        xfail_match("Unsupported operation.+stablehlo.reduce_window")
                    ],
                ),
                NumpyroDistributionTestConfig(
                    dists.Poisson,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                    differentiable_argnums=(1,),
                ),
                NumpyroDistributionTestConfig(
                    dists.GeometricProbs,
                    lambda rng, bs=batch_shape: rng.uniform(0.1, 0.9, bs),
                    differentiable_argnums=(1,),
                    name="Geometric",
                ),
                NumpyroDistributionTestConfig(
                    dists.NegativeBinomial2,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # mean
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # concentration
                    differentiable_argnums=(1, 2),
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.MultinomialProbs,
                        lambda rng, bs=batch_shape: rng.dirichlet(numpy.ones(5), bs),
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
                    lambda rng, bs=batch_shape: rng.standard_normal(bs),
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Logistic,
                    lambda rng, bs=batch_shape: rng.standard_normal(bs),
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),
                ),
                NumpyroDistributionTestConfig(
                    dists.Pareto,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # scale
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # alpha
                ),
                NumpyroDistributionTestConfig(
                    dists.Weibull,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # scale
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # concentration
                ),
                NumpyroDistributionTestConfig(
                    dists.Chi2,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs) + 2,  # df
                ),
                NumpyroDistributionTestConfig(
                    dists.InverseGamma,
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # concentration
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # rate
                ),
                NumpyroDistributionTestConfig(
                    dists.VonMises,
                    lambda rng, bs=batch_shape: rng.uniform(
                        -numpy.pi, numpy.pi, bs
                    ),  # loc
                    lambda rng, bs=batch_shape: rng.gamma(5, 5, bs),  # concentration
                ),
                # Multivariate distributions.
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.MultivariateNormal,
                        lambda rng, bs=batch_shape: rng.standard_normal(
                            bs + (4,)
                        ),  # loc
                        None,  # covariance_matrix
                        None,  # precision_matrix
                        lambda rng: numpy.linalg.cholesky(
                            numpy.eye(4) + numpy.ones((4, 4))
                        ),
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
                        lambda rng, bs=batch_shape: rng.standard_normal(
                            bs + (4,)
                        ),  # loc
                        lambda rng, bs=batch_shape: rng.standard_normal(
                            bs + (4, 2)
                        ),  # cov_factor
                        lambda rng, bs=batch_shape: rng.gamma(
                            5, 5, bs + (4,)
                        ),  # cov_diag
                    ),
                    marks=[xfail_match("scatter:.+unsupported scatter pattern")],
                ),
            ]
