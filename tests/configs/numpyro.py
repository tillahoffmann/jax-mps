import numpy
import pytest
from jax import random
from numpyro import distributions as dists

from .util import OperationTestConfig


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
            lambda: dist_cls(*args).sample(random.key(17)),  # pyright: ignore[reportArgumentType]
            *args,
            **kwargs,
        )


def make_numpyro_op_configs():
    with OperationTestConfig.module_name("numpyro"):
        for batch_shape in [(), (3,)]:
            yield from [
                NumpyroDistributionTestConfig(
                    dists.Normal,
                    numpy.random.standard_normal(batch_shape),
                    numpy.random.gamma(5, 5, batch_shape),
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.Gamma,
                        numpy.random.gamma(5, 5, batch_shape),
                        numpy.random.gamma(5, 5, batch_shape),
                    ),
                    marks=[
                        pytest.mark.xfail(
                            reason="Sampling from gamma distribution requires 'stablehlo.while'.",
                            strict=True,
                        )
                    ],
                ),
                NumpyroDistributionTestConfig(
                    dists.Exponential,
                    numpy.random.gamma(5, 5, batch_shape),
                ),
                NumpyroDistributionTestConfig(
                    dists.Uniform,
                    numpy.random.standard_normal(batch_shape),
                    numpy.random.standard_normal(batch_shape) + 2,
                ),
                NumpyroDistributionTestConfig(
                    dists.Laplace,
                    numpy.random.standard_normal(batch_shape),
                    numpy.random.gamma(5, 5, batch_shape),
                ),
                NumpyroDistributionTestConfig(
                    dists.Cauchy,
                    numpy.random.standard_normal(batch_shape),
                    numpy.random.gamma(5, 5, batch_shape),
                ),
                NumpyroDistributionTestConfig(
                    dists.HalfNormal,
                    numpy.random.gamma(5, 5, batch_shape),
                ),
                NumpyroDistributionTestConfig(
                    dists.HalfCauchy,
                    numpy.random.gamma(5, 5, batch_shape),
                ),
                NumpyroDistributionTestConfig(
                    dists.LogNormal,
                    numpy.random.standard_normal(batch_shape),
                    numpy.random.gamma(5, 5, batch_shape),
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.Beta,
                        numpy.random.gamma(5, 5, batch_shape),
                        numpy.random.gamma(5, 5, batch_shape),
                    ),
                    marks=[
                        pytest.mark.xfail(
                            reason="Sampling from beta distribution requires 'stablehlo.while'.",
                            strict=True,
                        )
                    ],
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.StudentT,
                        numpy.random.gamma(5, 5, batch_shape) + 2,  # df > 2
                        numpy.random.standard_normal(batch_shape),
                        numpy.random.gamma(5, 5, batch_shape),
                    ),
                    marks=[
                        pytest.mark.xfail(
                            reason="Sampling from StudentT distribution requires 'stablehlo.while'.",
                            strict=True,
                        )
                    ],
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.Dirichlet,
                        numpy.random.gamma(5, 5, batch_shape + (3,)),
                    ),
                    marks=[
                        pytest.mark.xfail(
                            reason="Sampling from Dirichlet distribution requires 'stablehlo.while'.",
                            strict=True,
                        )
                    ],
                ),
                # Discrete distributions (differentiable_argnums excludes sample at 0).
                NumpyroDistributionTestConfig(
                    dists.BernoulliProbs,
                    numpy.random.uniform(0.1, 0.9, batch_shape),
                    differentiable_argnums=(1,),
                    name="Bernoulli",
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.BinomialProbs,
                        numpy.random.uniform(0.1, 0.9, batch_shape),
                        10,  # total_count (not differentiable)
                        differentiable_argnums=(1,),
                        name="Binomial",
                    ),
                    marks=[
                        pytest.mark.xfail(
                            reason="Sampling from Binomial requires 'stablehlo.while'.",
                            strict=True,
                        )
                    ],
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.CategoricalProbs,
                        numpy.random.dirichlet(numpy.ones(5), batch_shape),
                        differentiable_argnums=(1,),
                        name="Categorical",
                    ),
                    marks=[
                        pytest.mark.xfail(
                            reason="Sampling from Categorical requires 'stablehlo.while'.",
                            strict=True,
                        )
                    ],
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.Poisson,
                        numpy.random.gamma(5, 5, batch_shape),
                        differentiable_argnums=(1,),
                    ),
                    marks=[
                        pytest.mark.xfail(
                            reason="Sampling from Poisson requires 'stablehlo.while'.",
                            strict=True,
                        )
                    ],
                ),
                NumpyroDistributionTestConfig(
                    dists.GeometricProbs,
                    numpy.random.uniform(0.1, 0.9, batch_shape),
                    differentiable_argnums=(1,),
                    name="Geometric",
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.NegativeBinomial2,
                        numpy.random.gamma(5, 5, batch_shape),  # mean
                        numpy.random.gamma(5, 5, batch_shape),  # concentration
                        differentiable_argnums=(1, 2),
                    ),
                    marks=[
                        pytest.mark.xfail(
                            reason="Sampling from NegativeBinomial2 requires 'stablehlo.while'.",
                            strict=True,
                        )
                    ],
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.MultinomialProbs,
                        numpy.random.dirichlet(numpy.ones(5), batch_shape),
                        10,  # total_count (not differentiable)
                        differentiable_argnums=(1,),
                        name="Multinomial",
                    ),
                    marks=[
                        pytest.mark.xfail(
                            reason="Sampling from Multinomial requires 'stablehlo.while'.",
                            strict=True,
                        )
                    ],
                ),
                # Additional continuous distributions.
                NumpyroDistributionTestConfig(
                    dists.Gumbel,
                    numpy.random.standard_normal(batch_shape),
                    numpy.random.gamma(5, 5, batch_shape),
                ),
                NumpyroDistributionTestConfig(
                    dists.Logistic,
                    numpy.random.standard_normal(batch_shape),
                    numpy.random.gamma(5, 5, batch_shape),
                ),
                NumpyroDistributionTestConfig(
                    dists.Pareto,
                    numpy.random.gamma(5, 5, batch_shape),  # scale
                    numpy.random.gamma(5, 5, batch_shape),  # alpha
                ),
                NumpyroDistributionTestConfig(
                    dists.Weibull,
                    numpy.random.gamma(5, 5, batch_shape),  # scale
                    numpy.random.gamma(5, 5, batch_shape),  # concentration
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.Chi2,
                        numpy.random.gamma(5, 5, batch_shape) + 2,  # df
                    ),
                    marks=[
                        pytest.mark.xfail(
                            reason="Sampling from Chi2 requires 'stablehlo.while'.",
                            strict=True,
                        )
                    ],
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.InverseGamma,
                        numpy.random.gamma(5, 5, batch_shape),  # concentration
                        numpy.random.gamma(5, 5, batch_shape),  # rate
                    ),
                    marks=[
                        pytest.mark.xfail(
                            reason="Sampling from InverseGamma requires 'stablehlo.while'.",
                            strict=True,
                        )
                    ],
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.VonMises,
                        numpy.random.uniform(-numpy.pi, numpy.pi, batch_shape),  # loc
                        numpy.random.gamma(5, 5, batch_shape),  # concentration
                    ),
                    marks=[
                        pytest.mark.xfail(
                            reason="Sampling from VonMises requires 'stablehlo.while'.",
                            strict=True,
                        )
                    ],
                ),
                # Multivariate distributions.
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.MultivariateNormal,
                        numpy.random.standard_normal(batch_shape + (4,)),  # loc
                        None,  # covariance_matrix
                        None,  # precision_matrix
                        numpy.linalg.cholesky(numpy.eye(4) + numpy.ones((4, 4))),
                    ),
                    marks=[
                        pytest.mark.xfail(
                            reason="MultivariateNormal requires 'stablehlo.triangular_solve'.",
                            strict=True,
                        )
                    ],
                ),
                pytest.param(
                    NumpyroDistributionTestConfig(
                        dists.LowRankMultivariateNormal,
                        numpy.random.standard_normal(batch_shape + (4,)),  # loc
                        numpy.random.standard_normal(
                            batch_shape + (4, 2)
                        ),  # cov_factor
                        numpy.random.gamma(5, 5, batch_shape + (4,)),  # cov_diag
                    ),
                    marks=[
                        pytest.mark.xfail(
                            reason="LowRankMultivariateNormal requires unsupported scatter pattern.",
                            strict=True,
                        )
                    ],
                ),
            ]
