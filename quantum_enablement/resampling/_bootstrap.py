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

"""Bootstrapping tools and techniques."""


from collections.abc import Mapping, Sequence
from functools import singledispatchmethod

from numpy import (  # pylint: disable=redefined-builtin
    abs,
    asarray,
    complex_,
    float_,
    sum,
)
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from qiskit.primitives import SamplerResult
from qiskit.result import ProbDistribution, QuasiDistribution

from ._typing import (
    QiskitSamplesLike,
    compose_results,
    distribution_to_quasi_dist,
    distribution_to_result,
    is_counts_like,
    is_numeric_array,
    is_qiskit_samples_like,
    samples_to_distribution,
)


# TODO: warn if requested size exceeds input size/shots
class Bootstrap:
    """Bootstrap resampling utility class.

    Args:
        default_size: the number of (re)samples to collect randomly
            in the absence of any other provided or inferred `size`.
            If `None`, it will default to `10_000`.
        seed: an optional seed for the internal (pseudo-)random number generator.
    """

    def __init__(self, *, default_size: int | None = None, seed: int | None = None) -> None:
        self.default_size = default_size
        self._rng: Generator = default_rng(seed)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        default_size = self.default_size
        return f"{cls_name}({default_size})"

    ################################## PROPERTIES ##################################
    @property
    def default_size(self) -> int:
        """Default resample size."""
        return self._default_size

    @default_size.setter
    def default_size(self, size: int | None) -> None:
        size = size or 10_000
        self._default_size: int = self._validate_size(size)

    ################################## PUBLIC API ##################################
    @singledispatchmethod
    def resample(self, samples, *, size: int | None = None) -> list:
        """Resample randomly.

        Args:
            samples: original `SamplesLike` object to resample from.
            size: the number of (re)samples to collect randomly.
                If `None`, it will try to be inferred from `samples`
                and fall back to the default value otherwise.

        Returns:
            List of randomly resampled elements.
        """
        raise TypeError(f"Expected type 'SamplesLike', got '{type(samples)}' instead.")

    @resample.register
    def _(self, samples: Sequence, *, size: int | None = None) -> list:
        if len(samples) == 0:
            raise ValueError("Empty input, no samples to resample from.")
        inferred_size = len(samples)
        size = size or inferred_size or self.default_size
        size = self._validate_size(size)
        resamples = self._rng.choice(samples, p=None, size=size, replace=True)
        return resamples.tolist()

    @resample.register
    def _(self, samples: Mapping, *, size: int | None = None) -> list:
        if len(samples) == 0:
            raise ValueError("Empty input, no samples to resample from.")
        population = list(samples.keys())
        frequencies = list(samples.values())
        if not is_numeric_array(frequencies):
            raise TypeError("Input sample frequencies are not of numeric type.")
        inferred_size = sum(frequencies) if is_counts_like(samples) else None
        size = size or inferred_size or self.default_size
        size = self._validate_size(size)
        probabilities = self._derive_probabilities(frequencies)
        resamples = self._rng.choice(population, p=probabilities, size=size, replace=True)
        return resamples.tolist()

    @resample.register
    def _(self, samples: ProbDistribution, *, size: int | None = None) -> list[int]:
        size = size or samples.shots
        mapping = dict(samples)
        return self.resample(mapping, size=size)

    @resample.register
    def _(self, samples: QuasiDistribution, *, size: int | None = None) -> list[int]:
        distribution = samples.nearest_probability_distribution(return_distance=False)
        return self.resample(distribution, size=size)

    @resample.register
    def _(self, samples: SamplerResult, *, size: int | None = None) -> list[int]:
        if samples.num_experiments == 0:
            raise ValueError("No experiments to resample from in input SamplerResult.")
        if samples.num_experiments > 1:
            raise ValueError(
                f"Number of experiments in input result ({samples.num_experiments}) "
                "exceeds the maximum number of experiments allowed (1)."
            )
        quasi_dist = samples.quasi_dists[0]
        metadatum = samples.metadata[0]
        size = size or metadatum.get("shots", None)
        return self.resample(quasi_dist, size=size)

    def resample_distribution(
        self, samples: QiskitSamplesLike, *, size: int | None = None
    ) -> ProbDistribution:
        """Resample probability distribution randomly.

        Args:
            samples: original `QiskitSamplesLike` object to resample from.
            size: the number of (re)samples to collect randomly.
                If `None`, it will try to be inferred from `samples`
                and fall back to the default value otherwise.

        Returns:
            ProbDistribution object randomly resampled from input samples.
        """
        if not is_qiskit_samples_like(samples):
            raise TypeError(f"Expected type 'QiskitSamplesLike', got '{type(samples)}' instead.")
        memory = self.resample(samples, size=size)
        return samples_to_distribution(memory)

    def resample_quasi_dist(
        self, samples: QiskitSamplesLike, *, size: int | None = None
    ) -> QuasiDistribution:
        """Resample quasi-distribution randomly.

        Args:
            samples: original `QiskitSamplesLike` object to resample from.
            size: the number of (re)samples to collect randomly.
                If `None`, it will try to be inferred from `samples`
                and fall back to the default value otherwise.

        Returns:
            QuasiDistribution object randomly resampled from input samples.
        """
        distribution = self.resample_distribution(samples, size=size)
        return distribution_to_quasi_dist(distribution)

    @singledispatchmethod
    def resample_result(self, samples, *, size: int | None = None) -> SamplerResult:
        """Resample sampler result randomly.

        Args:
            samples: original `QiskitSamplesLike` object to resample from.
            size: the number of (re)samples to collect randomly.
                If `None`, it will try to be inferred from `samples`
                and fall back to the default value otherwise.

        Returns:
            SamplerResult object randomly resampled from input samples.
        """
        distribution = self.resample_distribution(samples, size=size)
        return distribution_to_result(distribution)

    @resample_result.register
    def _(self, samples: SamplerResult, *, size: int | None = None) -> SamplerResult:
        if samples.num_experiments > 1:
            results = (self.resample_result(result) for result in samples.decompose())
            return compose_results(*results)
        distribution = self.resample_distribution(samples, size=size)
        result = distribution_to_result(distribution)
        result.metadata[0] = {**samples.metadata[0], **result.metadata[0]}
        return result

    ################################## VALIDATION ##################################
    def _validate_size(self, size: int) -> int:
        """Validate input size."""
        try:
            size = int(size)
        except (TypeError, ValueError) as error:
            raise TypeError(f"Expected type 'int', got '{type(size)}' instead.") from error
        if size < 1:
            raise ValueError(f"Resampling size must be greater than zero, got '{size}' instead.")
        return size

    def _derive_probabilities(
        self, coefficients: Sequence[complex] | NDArray[complex_]
    ) -> NDArray[float_]:
        """Derives probabilities from a sequence of complex coefficients using the L1 norm."""
        coefficients = asarray(coefficients)
        if not is_numeric_array(coefficients):
            raise TypeError("Input coefficients are not of numeric type.")
        weights = abs(coefficients)
        probabilities = weights / sum(weights)
        return probabilities
