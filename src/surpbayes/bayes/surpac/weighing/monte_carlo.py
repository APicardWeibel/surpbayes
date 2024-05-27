r"""
Sub module for weighing a sample of points in order to approximate a distribution.

Methods in the Bayesian calibration module rely on estimation of integrals with respect to
variable distributions $\pi(\theta)$.

These integrals typically involve a function $f$ which is costly to evaluate. Previous evaluations
of $f$ at a fixed sample $(x_i)$ is available.

The goal of the present module is to compute weights $(\omega_i)$, such that
    $$\sum g(x_i)\omega_i \simeq  \pi(\theta)[g] $$
for all function $g$.

The weights approximated here are defined thus:
For sample $x_i$, consider partition $S(x_i) = \{x, i \in \arg\min_j (d(x,x_j))\}$.
Consider $\omega_i = \pi_\theta[S(x_i)]$.

For standard distribution, the distance $d$ used is the standard 2 norm between vectors.
For Gaussian distribution, the distance $d$ used is modified 2 norm based on scalar product

$$\langle a, b \rangle  = a^t \cdot cov^{-1} b.$$

The choice of weights $\omega_i$ amounts to replacing $g$ by a 1NN approximation for the integral.

In practice, computation of weights $\omega_i$ as defined above is difficult, as it involves
integrating the density on polygons. The weights $\omega_i$ are therefore approximated by a Monte
Carlo procedure. A sample $\tilde{x}_j$ is generated from distribution $\pi(\theta)$, and
$\hat{\omega_i}$ is defined as a fraction of sample $\tilde{x}_j$ having closest neighbor $x_i$.

These approximated weights are efficiently computed using KD Trees (scipy.spatial implementation).
"""

# TODO: improve interface with Faiss Indexes logic!

from typing import Optional

import faiss
import numpy as np
from surpbayes.misc import prod
from surpbayes.proba import Gaussian, Proba
from surpbayes.types import Samples


def get_weights_mc(
    proba: Proba,
    samples: Samples,
    metric_matrix: Optional[np.ndarray] = None,
    half_metric_matrix: Optional[np.ndarray] = None,
    n_sample_estim: int = 10**6,
    k_neighbors=1,
) -> np.ndarray:
    """Approximate weight given by 'proba' to the Voronoi partition of 'samples'. The distance
    used to compute the Voronoi partition is

        $$ d(x,y)^2 = (x-y)^T metric_matrix (x-y) = \lVert half_metric_matrix @ (x-y)\rVert^2 $$

    Approximation is performed by sampling "n_sample_estim" points from "proba" and finding nearest
    point in "samples" (amounts to a standard monte carlo estimation of volume). Exact Neighbor
    search is performed by library faiss.
    """
    large_sample = proba(n_sample_estim)
    if len(proba.sample_shape) >= 2:
        large_sample = large_sample.reshape((n_sample_estim, prod(proba.sample_shape)))
        samples = samples.reshape((samples[0], prod(proba.sample_shape)))

    if (half_metric_matrix is None) & (metric_matrix is not None):
        vals, vects = np.linalg.eigh(metric_matrix)  # type: ignore
        half_metric_matrix = (np.sqrt(vals) * vects) @ vects.T

    d = proba._sample_size  # pylint: disable=protected-access

    if half_metric_matrix is not None:
        index = faiss.IndexFlatL2(d)
        index.add(
            (samples @ half_metric_matrix).astype(np.float32)
        )  # pylint: disable=no-value-for-parameter
        min_ids = index.search(
            (large_sample @ half_metric_matrix).astype(np.float32), k=k_neighbors
        )[
            1
        ]  # pylint: disable=no-value-for-parameter
    else:
        index = faiss.IndexFlatL2(d)
        index.add(samples.astype(np.float32))  # pylint: disable=no-value-for-parameter
        min_ids = index.search(large_sample.astype(np.float32), k=k_neighbors)[
            1
        ]  # pylint: disable=no-value-for-parameter

    idx, counts = np.unique(min_ids, return_counts=True)

    # Some indexes can be negative if faiss fails!
    # Error handling for this case
    if np.any((idx < 0) | (idx >= len(samples))):

        raise Exception("Faiss wrong idx Error")

    weights = np.zeros(len(samples))
    weights[idx] = counts / (n_sample_estim * k_neighbors)

    return weights


def get_weights_mc_approx(
    proba: Proba,
    samples: Samples,
    metric_matrix: Optional[np.ndarray] = None,
    half_metric_matrix: Optional[np.ndarray] = None,
    n_sample_estim: int = 10**6,
    k_neighbors=1,
    nlist=100,
    nprobe=4,
) -> np.ndarray:
    """Approximate weight given by 'proba' to the Voronoid partition of 'samples'. The distance
    used to compute the Voronoid partition is

        $$ d(x,y)^2 = (x-y)^T metric_matrix (x-y) = \lVert half_metric_matrix @ (x-y)\rVert^2 $$

    Approximation is performed by sampling "n_sample_estim" points from "proba" and finding nearest
    point in "samples" (amounts to a standard monte carlo estimation of volume). Approximate Neighbor
    search is performed by library faiss using IndexIVFFlat
    """
    n_samples = len(samples)
    if n_samples < 40:
        # Not enough samples, use exact routine
        return get_weights_mc(
            proba=proba,
            samples=samples,
            metric_matrix=metric_matrix,
            half_metric_matrix=half_metric_matrix,
            n_sample_estim=n_sample_estim,
            k_neighbors=k_neighbors,
        )
    if 39 * nlist > len(samples):
        nlist = max(1, len(samples) // 39)

    if 4 * nprobe > nlist:
        nprobe = max(1, nlist // 4)

    large_sample = proba(n_sample_estim)
    if len(proba.sample_shape) >= 2:
        large_sample = large_sample.reshape((n_sample_estim, prod(proba.sample_shape)))
        samples = samples.reshape((samples[0], prod(proba.sample_shape)))

    if (half_metric_matrix is None) & (metric_matrix is not None):
        vals, vects = np.linalg.eigh(metric_matrix)  # type: ignore
        half_metric_matrix = (np.sqrt(vals) * vects) @ vects.T

    d = proba._sample_size  # pylint: disable=protected-access

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.nprobe = nprobe

    if half_metric_matrix is not None:
        samples_fmt = (samples @ half_metric_matrix).astype(np.float32)
        index.train(samples_fmt)
        index.add(samples_fmt)
        min_ids = index.search(
            (large_sample @ half_metric_matrix).astype(np.float32), k=k_neighbors
        )[1]
    else:
        samples_fmt = samples.astype(np.float32)
        index.train(samples_fmt)
        index.add(samples_fmt)
        min_ids = index.search(large_sample.astype(np.float32), k=k_neighbors)[1]

    idx, counts = np.unique(min_ids, return_counts=True)

    # Some indexes can be negative if faiss fails!
    # Error handling for this case
    if np.any((idx < 0) | (idx >= len(samples))):

        raise Exception(f"Faiss wrong idx Error")

    weights = np.zeros(len(samples))
    weights[idx] = counts / (n_sample_estim * k_neighbors)

    return weights


def get_weights_mc_gauss(
    proba: Gaussian,
    samples: Samples,
    n_sample_estim: int = 10**6,
    k_neighbors: int = 1,
) -> np.ndarray:
    """
    Obtain weights by approximating the mass of sets of attraction of each point for a gaussian
    distribution.

    Variant of get_weights function for gaussian. The distance metric is induced by the covariance.
    """
    vals, vects = proba.vals, proba.vects
    half_metric_matrix = ((1 / np.sqrt(vals)) * vects) @ vects.T
    return get_weights_mc(
        proba,
        samples,
        half_metric_matrix=half_metric_matrix,
        n_sample_estim=n_sample_estim,
        k_neighbors=k_neighbors,
    )


def get_weights_mc_approx_gauss(
    proba: Gaussian,
    samples: Samples,
    n_sample_estim: int = 10**6,
    k_neighbors: int = 1,
    nlist: int = 100,
    nprobe: int = 4,
) -> np.ndarray:
    """
    Obtain weights by approximating the mass of sets of attraction of each point for a gaussian
    distribution.

    Variant of get_weights function for gaussian. The distance metric is induced by the covariance.
    """
    vals, vects = proba.vals, proba.vects
    half_metric_matrix = ((1 / np.sqrt(vals)) * vects) @ vects.T

    return get_weights_mc_approx(
        proba,
        samples,
        half_metric_matrix=half_metric_matrix,
        n_sample_estim=n_sample_estim,
        k_neighbors=k_neighbors,
        nlist=nlist,
        nprobe=nprobe,
    )
