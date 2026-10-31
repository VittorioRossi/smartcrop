"""Tests for CDFRemaskingSampler — existing behaviour + bucket batching.

Uses a CPU-only MockLM so no GPU is needed.

MockLM is keyed on x[b, 0] (first token of each prompt):
  trigger_map[first_token] = gen_offset
  → positions >= prompt_len + gen_offset: p_eos is high (EOS logit = +10)
  → everywhere else: p_content is high (CONTENT_ID logit = 0, rest = -10)

This lets the model behave correctly even when bucket batching reorders inputs,
because the trigger is encoded in the sequence content itself.
"""

import torch
from dataclasses import dataclass

from diffusion_llms.generators.cdf_remasking_generator import (
    _predict_lengths_from_t1,
    CDFRemaskingSampler,
    CDFRemaskingSamplerConfig,
)
from diffusion_llms.generators.cdf_remasking_batched import CDFRemaskingSamplerBatched

VOCAB      = 64
EOS_ID     = 1
MASK_ID    = 2
CONTENT_ID = 3
BOS_ID     = 0
HIGH       = 10.0
LOW        = -10.0


# ── Shared helpers ────────────────────────────────────────────────────────────

@dataclass
class _Out:
    logits: torch.Tensor


class MockLM(torch.nn.Module):
    """Deterministic CPU mock LM.

    Keyed on x[b, 0]: EOS logit is HIGH at positions >= prompt_len + trigger_map[first_token].
    Content token wins everywhere else.
    """

    def __init__(self, prompt_len: int, trigger_map: dict[int, int]):
        super().__init__()
        self.prompt_len  = prompt_len
        self.trigger_map = trigger_map
        self._dev        = torch.device("cpu")

    @property
    def device(self):
        return self._dev

    def forward(self, x, attention_mask=None):
        B, T = x.shape
        logits = torch.full((B, T, VOCAB), LOW)
        logits[:, :, CONTENT_ID] = 0.0
        for b in range(B):
            first_tok   = x[b, 0].item()
            offset      = self.trigger_map.get(first_tok, T)
            abs_trigger = self.prompt_len + offset
            if abs_trigger < T:
                logits[b, abs_trigger:, EOS_ID] = HIGH
        return _Out(logits)


class MockTokenizer:
    eos_token_id  = EOS_ID
    mask_token_id = MASK_ID
    bos_token_id  = BOS_ID


def prompt(first_token: int, length: int) -> list[int]:
    """Prompt whose identity is encoded in its first token."""
    return [first_token] + [5] * (length - 1)


def make_sampler(prompt_len: int, trigger_map: dict[int, int]) -> CDFRemaskingSampler:
    return CDFRemaskingSampler(
        model=MockLM(prompt_len, trigger_map),
        tokenizer=MockTokenizer(),
    )


def make_batched_sampler(prompt_len: int, trigger_map: dict[int, int]) -> CDFRemaskingSamplerBatched:
    return CDFRemaskingSamplerBatched(
        model=MockLM(prompt_len, trigger_map),
        tokenizer=MockTokenizer(),
    )


def make_cfg(max_new_tokens: int = 30, steps: int = 1) -> CDFRemaskingSamplerConfig:
    return CDFRemaskingSamplerConfig(
        max_new_tokens=max_new_tokens,
        steps=steps,
        block_size=max_new_tokens,   # one block → everything revealed in 1 step
        cdf_quantile=0.90,
        eos_free=False,
        temperature=0.0,
    )


# ── _predict_lengths_from_t1 (existing, must not regress) ────────────────────

class TestPredictLengths:

    def test_crossing_at_various_offsets(self):
        pl = 10
        model  = MockLM(pl, {10: 5, 11: 12})
        inputs = [torch.tensor(prompt(10, pl)),   # crossing at gen pos 5 → pred_len 6
                  torch.tensor(prompt(11, pl))]   # crossing at gen pos 12 → pred_len 13
        pred_lens = _predict_lengths_from_t1(model, inputs, MASK_ID, EOS_ID, 20, quantile=0.90)
        assert pred_lens == [6, 13], pred_lens

    def test_no_crossing_returns_max_new_tokens(self):
        pl    = 5
        model = MockLM(pl, {})                   # no trigger → p_eos never >= 0.90
        pred_lens = _predict_lengths_from_t1(
            model, [torch.tensor(prompt(10, pl))], MASK_ID, EOS_ID, 15, quantile=0.90
        )
        assert pred_lens == [15]

    def test_crossing_at_position_zero(self):
        pl    = 5
        model = MockLM(pl, {10: 0})              # EOS high from gen pos 0 → pred_len 1
        pred_lens = _predict_lengths_from_t1(
            model, [torch.tensor(prompt(10, pl))], MASK_ID, EOS_ID, 10, quantile=0.90
        )
        assert pred_lens == [1]

    def test_batch_of_three(self):
        pl    = 8
        model = MockLM(pl, {10: 3, 11: 8, 12: 200})
        inputs = [torch.tensor(prompt(ft, pl)) for ft in [10, 11, 12]]
        pred_lens = _predict_lengths_from_t1(model, inputs, MASK_ID, EOS_ID, 20, quantile=0.90)
        assert pred_lens == [4, 9, 20], pred_lens

    def test_quantile_boundary(self):
        """Crossing is detected only when p_eos strictly >= quantile."""
        pl    = 5
        model = MockLM(pl, {10: 7})              # crossing at offset 7 → pred_len 8
        pred_lens = _predict_lengths_from_t1(
            model, [torch.tensor(prompt(10, pl))], MASK_ID, EOS_ID, 20, quantile=0.90
        )
        assert pred_lens == [8]


# ── sample() — existing behaviour, must not regress ──────────────────────────

class TestExistingSample:

    def test_output_shape(self):
        """Output sequence = prompt tokens + canvas tokens."""
        pl      = 6
        sampler = make_sampler(pl, {10: 5})
        cfg     = make_cfg(max_new_tokens=20)
        out     = sampler.sample([prompt(10, pl)], cfg, return_dict=True)
        seq     = out.sequences[0]
        assert seq.shape[0] == pl + 5 + 1, (   # pred_len=6 → canvas=6
            f"Expected length {pl + 6}, got {seq.shape[0]}"
        )

    def test_prompt_tokens_preserved(self):
        """Prompt tokens at the front of the output must be unchanged."""
        pl      = 8
        p       = prompt(42, pl)
        sampler = make_sampler(pl, {42: 10})
        cfg     = make_cfg(max_new_tokens=20)
        out     = sampler.sample([p], cfg, return_dict=True)
        seq     = out.sequences[0].tolist()
        assert seq[:pl] == p, f"Prompt corrupted: {seq[:pl]} != {p}"

    def test_eos_free_suppresses_eos(self):
        """With eos_free=True, EOS must not appear in the generated region."""
        pl      = 6
        sampler = make_sampler(pl, {10: 5})
        cfg     = make_cfg(max_new_tokens=20)
        cfg.eos_free = True
        out  = sampler.sample([prompt(10, pl)], cfg, return_dict=True)
        seq  = out.sequences[0].tolist()
        gen  = seq[pl:]
        assert EOS_ID not in gen, f"EOS found in generation despite eos_free: {gen}"

    def test_canvas_cropped_to_pred_len(self):
        """Canvas must be cropped to pred_len, not left at max_new_tokens."""
        pl        = 6
        pred_len  = 7                            # trigger offset 6 → pred_len 7
        sampler   = make_sampler(pl, {10: 6})
        cfg       = make_cfg(max_new_tokens=30)
        out       = sampler.sample([prompt(10, pl)], cfg, return_dict=True)
        ntoken    = out.sequences[0].shape[0] - pl
        assert ntoken == pred_len, (
            f"Canvas should be {pred_len}, got {ntoken} (not cropped?)"
        )

    def test_batch_size_1_deterministic(self):
        """Calling sample() twice with temperature=0 gives identical output."""
        pl      = 6
        sampler = make_sampler(pl, {10: 8})
        cfg     = make_cfg(max_new_tokens=20, steps=1)
        p       = prompt(10, pl)
        seq1    = sampler.sample([p], cfg, return_dict=True).sequences[0].tolist()
        seq2    = sampler.sample([p], cfg, return_dict=True).sequences[0].tolist()
        assert seq1 == seq2


# ── sample_batched() — new behaviour ─────────────────────────────────────────

class TestSampleBatched:

    def test_returns_one_result_per_input(self):
        pl      = 6
        sampler = make_batched_sampler(pl, {10: 5, 11: 10, 12: 15})
        inputs  = [prompt(ft, pl) for ft in [10, 11, 12]]
        results = sampler.sample_batched(inputs, make_cfg(), batch_size=8)
        assert len(results) == 3

    def test_original_order_preserved(self):
        """Results must come back in input order even after bucket-sort."""
        pl          = 6
        trigger_map = {20: 20, 21: 2, 22: 14, 23: 5}   # very different lengths → will reorder
        sampler     = make_batched_sampler(pl, trigger_map)
        first_toks  = [20, 21, 22, 23]
        inputs      = [prompt(ft, pl) for ft in first_toks]

        results = sampler.sample_batched(inputs, make_cfg(max_new_tokens=30), batch_size=2)

        for i, ft in enumerate(first_toks):
            seq = results[i]["sequence"]
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()
            assert seq[0] == ft, (
                f"result[{i}] starts with {seq[0]}, expected {ft} — order not preserved"
            )

    def test_canvas_bounded_per_bucket(self):
        """With batch_size=1, each sample is its own bucket; ntoken == pred_len."""
        pl          = 6
        trigger_map = {10: 3, 11: 18}      # pred_lens: 4 and 19
        sampler     = make_batched_sampler(pl, trigger_map)

        results = sampler.sample_batched(
            [prompt(10, pl), prompt(11, pl)], make_cfg(max_new_tokens=30), batch_size=1
        )

        assert results[0]["ntoken"] == 4,  f"got {results[0]['ntoken']}"
        assert results[1]["ntoken"] == 19, f"got {results[1]['ntoken']}"

    def test_elapsed_non_negative(self):
        pl      = 6
        sampler = make_batched_sampler(pl, {10: 5})
        results = sampler.sample_batched([prompt(10, pl)], make_cfg(), batch_size=1)
        assert results[0]["elapsed"] >= 0

    def test_matches_sequential_sample(self):
        """sample_batched(batch_size=1) must produce identical sequences to sample()."""
        pl          = 8
        trigger_map = {30: 6, 31: 12, 32: 3}
        sampler     = make_batched_sampler(pl, trigger_map)
        first_toks  = [30, 31, 32]
        inputs      = [prompt(ft, pl) for ft in first_toks]
        cfg         = make_cfg(max_new_tokens=20, steps=1)

        seqs_seq = [
            sampler.sample([inp], cfg, return_dict=True).sequences[0].tolist()
            for inp in inputs
        ]
        results  = sampler.sample_batched(inputs, cfg, batch_size=1)
        seqs_bat = [r["sequence"].tolist() for r in results]

        for i in range(len(inputs)):
            assert seqs_seq[i] == seqs_bat[i], (
                f"Mismatch at index {i}:\n"
                f"  sequential: {seqs_seq[i]}\n"
                f"  batched:    {seqs_bat[i]}"
            )

    def test_result_keys(self):
        """Each result dict must have the keys run_benchmark.py depends on."""
        pl      = 6
        sampler = make_batched_sampler(pl, {10: 5})
        results = sampler.sample_batched([prompt(10, pl)], make_cfg(), batch_size=1)
        for key in ("sequence", "pred_len", "ntoken", "elapsed"):
            assert key in results[0], f"missing key '{key}'"
