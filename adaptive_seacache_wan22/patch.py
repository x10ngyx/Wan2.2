from __future__ import annotations

from functools import wraps

from wan.modules.model import WanModel


def patch_wan_model_forward_for_adaptive_seacache() -> None:
    """Inject raw latents into adaptive SeaCache before WanModel.forward runs.

    The main WanModel SeaCache hook receives transformer token features, while
    the adaptive predictor was trained on raw latent tensors. This wrapper keeps
    the main Wan code unchanged and gives the adaptive cache access to the same
    latent passed into the model for the current branch.
    """

    if getattr(WanModel.forward, "_adaptive_seacache_patched", False):
        return

    original_forward = WanModel.forward

    @wraps(original_forward)
    def wrapped_forward(self, x, *args, **kwargs):
        timestep_cache = kwargs.get("timestep_cache")
        cache_key = kwargs.get("timestep_cache_key")
        if (
            timestep_cache is not None
            and cache_key is not None
            and hasattr(timestep_cache, "set_current_latent")
        ):
            timestep_cache.set_current_latent(cache_key, x[0])
        return original_forward(self, x, *args, **kwargs)

    wrapped_forward._adaptive_seacache_patched = True
    WanModel.forward = wrapped_forward

