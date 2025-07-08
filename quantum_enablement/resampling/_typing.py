# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Typing definitions and tools for resampling."""


from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Union, get_args

from numpy import asarray
from numpy.typing import ArrayLike
from qiskit.primitives import SamplerResult
from qiskit.primitives.base.base_result import BasePrimitiveResult
from qiskit.result import ProbDistribution, QuasiDistribution

# TODO: add Memory (define class with casting methods)
# TODO: add Counts
QiskitSamplesLike = Union[ProbDistribution, QuasiDistribution, SamplerResult]
SamplesLike = Union[Sequence, Mapping, QiskitSamplesLike]


################################################################################
## CASTING
################################################################################
def samples_to_distribution(samples: Sequence) -> ProbDistribution:
    """Cast samples into probability distribution object."""
    if not isinstance(samples, Sequence):
        raise TypeError(f"Expected type 'Sequence', got '{type(samples)}' instead.")
    counts = Counter(samples)
    size = len(samples)
    distribution = {key: value / size for key, value in counts.items()}
    return ProbDistribution(distribution, shots=size)


def samples_to_quasi_dist(samples: Sequence) -> QuasiDistribution:
    """Cast samples to quasi-distribution object."""
    distribution = samples_to_distribution(samples)
    return distribution_to_quasi_dist(distribution)


def samples_to_result(samples: Sequence) -> SamplerResult:
    """Cast samples to sampler result object."""
    quasi_dist = samples_to_quasi_dist(samples)
    return quasi_dist_to_result(quasi_dist)


def distribution_to_quasi_dist(distribution: ProbDistribution) -> QuasiDistribution:
    """Cast probability distribution to quasi-distribution object."""
    if not isinstance(distribution, ProbDistribution):
        raise TypeError(f"Expected type 'ProbDistribution', got '{type(distribution)}' instead.")
    return QuasiDistribution(
        distribution,
        shots=distribution.shots,
        stddev_upper_bound=None,  # TODO: compute stddev_upper_bound
    )  # TODO: add class-method from ProbDistribution to qiskit


def distribution_to_result(distribution: ProbDistribution) -> SamplerResult:
    """Cast probability distribution to sampler result object."""
    quasi_dist = distribution_to_quasi_dist(distribution)
    return quasi_dist_to_result(quasi_dist)


def quasi_dist_to_result(quasi_dist: QuasiDistribution) -> SamplerResult:
    """Cast quasi-distribution to sampler result object."""
    if not isinstance(quasi_dist, QuasiDistribution):
        raise TypeError(f"Expected type 'QuasiDistribution', got '{type(quasi_dist)}' instead.")
    metadatum: dict[str, Any] = {}
    if quasi_dist.shots:
        metadatum["shots"] = quasi_dist.shots
    if quasi_dist.stddev_upper_bound:
        pass  # TODO: add more metadata
    return SamplerResult(quasi_dists=[quasi_dist], metadata=[metadatum])


################################################################################
## CHECKING
################################################################################
def is_numeric_array(a: ArrayLike, /) -> bool:
    """Check if array-like is over numeric types."""
    return asarray(a).dtype.kind in {"b", "u", "i", "f", "c"}


def is_qiskit_samples_like(o: Any, /) -> bool:
    """Check if input object is instance of QiskitSamplesLike."""
    types = get_args(QiskitSamplesLike)
    return isinstance(o, types)


def is_counts_like(o: Any, /) -> bool:
    """Check if input object is a counts-like (i.e. a multiset)."""
    if not isinstance(o, Mapping):
        return False
    counts = list(o.values())
    if not asarray(counts).dtype.kind in {"b", "u", "i"}:
        return False
    if not all(isinstance(c, int) for c in counts):
        return False
    return True


################################################################################
## MISC
################################################################################
# TODO: add to qiskit.primitives.base.base_result
def compose_results(*results) -> BasePrimitiveResult:
    """Compose primitive results into a single result."""
    result_type = type(results[0])
    if not issubclass(result_type, BasePrimitiveResult):
        raise TypeError(f"Expected type 'BasePrimitiveResult', got '{result_type}' instead.")
    if not all(isinstance(result, result_type) for result in results):
        raise TypeError(f"Type mismatch, expected all inputs to be of type '{result_type}'")
    experiment_values = (
        experiment.values()
        for result in results
        for experiment in result._generate_experiments()  # pylint: disable=protected-access
    )
    return result_type(*zip(*experiment_values))
