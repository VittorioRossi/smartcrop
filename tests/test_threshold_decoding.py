"""Tests for confidence-threshold parallel decoding in CDFRemaskingSampler.

CPU-only mocks. The t=1 probe is driven by MockLM's EOS trigger (see
test_cdf_batching.py); denoising confidence is controlled per mock below.
"""

from copy import copy

import torch

from dllm.core.samplers import MDLMSampler
from diffusion_llms.generators.cdf_remasking_generator import (
    CDFRemaskingSampler,
    CDFRemaskingSamplerConfig,
)
from tests.test_cdf_batching import (
    CONTENT_ID, EOS_ID, MASK_ID, VOCAB, MockLM, MockTokenizer, _Out, prompt,
)


class CountingLM(MockLM):
    """MockLM that counts forward calls."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls = 0

    def forward(self, x, attention_mask=None):
        self.calls += 1
        return super().forward(x, attention_mask)


class LowConfLM(CountingLM):
    """Like MockLM, but CONTENT_ID and a runner-up token tie closely at masked
    positions, so confidence is ~0.5 and never clears a 0.9 threshold."""

    def forward(self, x, attention_mask=None):
        out = super().forward(x, attention_mask)
        masked = x == MASK_ID
        out.logits[..., CONTENT_ID + 1] = torch.where(masked, -0.01, out.logits[..., CONTENT_ID + 1])
        # Ordered confidence: position p gets a bonus so the fallback order is left to right.
        T = x.shape[1]
        bonus = torch.linspace(0.5, 0.0, T)
        out.logits[..., CONTENT_ID] += torch.where(masked, bonus, torch.zeros_like(bonus))
        return out


class MaskArgmaxLM(CountingLM):
    """Predicts the mask token with top probability everywhere it is masked."""

    def forward(self, x, attention_mask=None):
        out = super().forward(x, attention_mask)
        if self.calls > 1:  # leave the t=1 probe untouched
            out.logits[..., MASK_ID] = torch.where(x == MASK_ID, 5.0, out.logits[..., MASK_ID])
        return out


def _cfg(max_new_tokens=40, block_size=8, threshold=0.9, steps=40):
    return CDFRemaskingSamplerConfig(
        max_new_tokens=max_new_tokens, steps=steps, block_size=block_size,
        cdf_quantile=0.90, temperature=0.0, threshold=threshold,
    )


def _sample(model, p, cfg):
    sampler = CDFRemaskingSampler(model=model, tokenizer=MockTokenizer())
    return sampler.sample([p], cfg, return_dict=True).sequences[0]


def test_threshold_none_matches_mdlm_bit_for_bit():
    """threshold=None must equal MDLMSampler on the cropped canvas."""
    pl, offset = 6, 20          # pred_len = 21, three blocks of 8
    p = prompt(10, pl)
    seq = _sample(LowConfLM(pl, {10: offset}), p, _cfg(threshold=None))

    cropped = copy(_cfg(threshold=None))
    cropped.max_new_tokens = offset + 1
    cropped.steps = offset + 1
    ref_sampler = MDLMSampler(model=LowConfLM(pl, {10: offset}), tokenizer=MockTokenizer())
    ref = ref_sampler.sample([p], cropped, return_dict=True).sequences[0]
    assert torch.equal(seq, ref)


def test_high_confidence_commits_whole_block_per_step():
    pl, offset = 6, 20          # pred_len 21 → blocks of 8, 8, 5
    model = CountingLM(pl, {10: offset})
    seq = _sample(model, prompt(10, pl), _cfg())
    assert not (seq == MASK_ID).any()
    assert seq.shape[0] == pl + offset + 1
    assert model.calls == 1 + 3  # probe + one step per block


def test_fallback_commits_one_token_per_step():
    pl, offset = 6, 20
    model = LowConfLM(pl, {10: offset})
    seq = _sample(model, prompt(10, pl), _cfg())
    assert not (seq == MASK_ID).any()
    assert model.calls == 1 + (offset + 1)  # probe + one token per step


def test_fallback_skips_mask_prediction():
    """When every masked position predicts the mask token, decoding still progresses."""
    pl, offset = 6, 10
    model = MaskArgmaxLM(pl, {10: offset})
    seq = _sample(model, prompt(10, pl), _cfg())
    gen = seq[pl:]
    assert not (gen == MASK_ID).any()
    assert model.calls == 1 + (offset + 1)


def test_eos_free_with_threshold():
    pl = 6
    cfg = _cfg()
    cfg.eos_free = True
    seq = _sample(CountingLM(pl, {10: 5}), prompt(10, pl), cfg)
    assert EOS_ID not in seq[pl:].tolist()


def test_block_size_larger_than_canvas():
    pl, offset = 6, 4
    model = CountingLM(pl, {10: offset})
    seq = _sample(model, prompt(10, pl), _cfg(block_size=32))
    assert not (seq == MASK_ID).any()
    assert model.calls == 2


def test_no_crop_skips_probe_and_uses_full_canvas():
    pl, offset, max_new = 6, 20, 40   # probe would crop to 21; no_crop keeps all 40
    model = CountingLM(pl, {10: offset})
    cfg = _cfg(max_new_tokens=max_new)
    cfg.no_crop = True
    seq = _sample(model, prompt(10, pl), cfg)
    assert seq.shape[0] == pl + max_new
    assert not (seq == MASK_ID).any()
    assert model.calls == max_new // 8  # one step per block, no probe
