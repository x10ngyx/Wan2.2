import types

import torch


def install_taylorseer(model, cache, stage_key):
    """Patch a WanModel instance with official-style TaylorSeer block forwards."""
    model._taylorseer_cache = cache
    model._taylorseer_stage_key = stage_key
    model._taylorseer_stream = None
    model._taylorseer_step_type = None
    attach_parent(model)
    for layer_index, block in enumerate(model.blocks):
        block._taylorseer_layer_index = layer_index
        block.forward = types.MethodType(taylorseer_block_forward, block)
    return model


def set_taylorseer_context(model, stream, step_type):
    model._taylorseer_stream = stream
    model._taylorseer_step_type = step_type


def taylorseer_block_forward(
    self,
    x,
    e,
    seq_lens,
    grid_sizes,
    freqs,
    context,
    context_lens,
    bwcache_state=None,
    bwcache_block_index=None,
    bwcache_cal_list_updated=True,
    bwcache_is_first_step=True,
    bwcache_metric="pooled_rel_l1",
):
    del bwcache_state
    del bwcache_block_index
    del bwcache_cal_list_updated
    del bwcache_is_first_step
    del bwcache_metric

    parent = _find_patched_parent(self)
    cache = parent._taylorseer_cache
    stage_key = parent._taylorseer_stage_key
    stream = parent._taylorseer_stream
    step_type = parent._taylorseer_step_type
    layer_index = self._taylorseer_layer_index

    if stream is None or step_type is None:
        raise ValueError("TaylorSeer context was not set before block forward.")

    assert e.dtype == torch.float32
    with torch.amp.autocast('cuda', dtype=torch.float32):
        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
    assert e[0].dtype == torch.float32

    if step_type == "Taylor":
        sa = cache.predict_module(stage_key, stream, layer_index, "self-attention")
        ca = cache.predict_module(stage_key, stream, layer_index, "cross-attention")
        ffn = cache.predict_module(stage_key, stream, layer_index, "ffn")
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + sa * e[2].squeeze(2)
        x = x + ca
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + ffn * e[5].squeeze(2)
        return x

    if step_type != "full":
        raise ValueError(f"Unsupported TaylorSeer step type: {step_type}")

    x_m = self._modulated_norm1(x, e)
    y = self.self_attn(x_m, seq_lens, grid_sizes, freqs)
    cache.record_module(stage_key, stream, layer_index, "self-attention", y)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        x = x + y * e[2].squeeze(2)

    y = self.cross_attn(self.norm3(x), context, context_lens)
    cache.record_module(stage_key, stream, layer_index, "cross-attention", y)
    x = x + y

    y = self.ffn(
        self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
    cache.record_module(stage_key, stream, layer_index, "ffn", y)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        x = x + y * e[5].squeeze(2)
    return x


def _find_patched_parent(block):
    parent = getattr(block, "_taylorseer_parent", None)
    if parent is not None:
        return parent
    raise ValueError("TaylorSeer block parent is not set.")


def attach_parent(model):
    for block in model.blocks:
        block._taylorseer_parent = model
