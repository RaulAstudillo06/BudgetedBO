from typing import Optional, Tuple

import torch
from botorch.posteriors import Posterior
from botorch.sampling.samplers import MCSampler, _check_shape_changed
from botorch.utils.sampling import manual_seed
from torch import Tensor

class PosteriorMeanSampler(MCSampler):
    r"""Sampler for MC base samples using iid N(0,1) samples.
    """

    def __init__(
        self,
        collapse_batch_dims: bool = True,
        batch_range: Tuple[int, int] = (0, -2),
    ) -> None:
        r"""Sampler for MC base samples using iid `N(0,1)` samples.
        Args:
            num_samples: The number of samples to use.
            resample: If `True`, re-draw samples in each `forward` evaluation -
                this results in stochastic acquisition functions (and thus should
                not be used with deterministic optimization algorithms).
            seed: The seed for the RNG. If omitted, use a random seed.
            collapse_batch_dims: If True, collapse the t-batch dimensions to
                size 1. This is useful for preventing sampling variance across
                t-batches.
            batch_range: The range of t-batch dimensions in the `base_sample_shape`
                used by `collapse_batch_dims`. The t-batch dims are
                batch_range[0]:batch_range[1]. By default, this is (0, -2),
                for the case where the non-batch dimensions are -2 (q) and
                -1 (d) and all dims in the front are t-batch dims.
        """
        super().__init__(batch_range=batch_range)
        self._sample_shape = torch.Size([1])
        self.collapse_batch_dims = collapse_batch_dims

    def _construct_base_samples(self, posterior: Posterior, shape: torch.Size) -> None:
        r"""Generate iid `N(0,1)` base samples (if necessary).
        This function will generate a new set of base samples and set the
        `base_samples` buffer if one of the following is true:
        - `resample=True`
        - the MCSampler has no `base_samples` attribute.
        - `shape` is different than `self.base_samples.shape` (if
            `collapse_batch_dims=True`, then batch dimensions of will be
            automatically broadcasted as necessary). This shape is expected to
            be `sample_shape + base_sample_shape`, where `base_sample_shape` has been
            adjusted to account for `collapse_batch_dims` (i.e., the output
            of the function `_get_base_sample_shape`).
        Args:
            posterior: The Posterior for which to generate base samples.
            shape: The shape of the base samples to construct.
        """
        if (_check_shape_changed(self.base_samples, self.batch_range, shape)
            or (not self.collapse_batch_dims and shape != self.base_samples.shape)
        ):
            base_samples = torch.zeros(shape, device=posterior.device, dtype=posterior.dtype)
            self.register_buffer("base_samples", base_samples)
        elif self.collapse_batch_dims and shape != self.base_samples.shape:
            self.base_samples = self.base_samples.view(shape)
        if self.base_samples.device != posterior.device:
            self.to(device=posterior.device)  # pragma: nocover
        if self.base_samples.dtype != posterior.dtype:
            self.to(dtype=posterior.dtype)