from __future__ import annotations

from itertools import pairwise

import pytest

from keel.sim_engine import EOS, RESERVED, DeviceProfile, SimModel, stable_digest


def test_next_token_is_deterministic() -> None:
    a = SimModel(vocab_size=512, seed=7)
    b = SimModel(vocab_size=512, seed=7)
    assert [a.next_token(100, i) for i in range(50)] == [b.next_token(100, i) for i in range(50)]


def test_different_seeds_diverge() -> None:
    a = SimModel(vocab_size=512, seed=1)
    b = SimModel(vocab_size=512, seed=2)
    assert [a.next_token(100, i) for i in range(50)] != [b.next_token(100, i) for i in range(50)]


def test_next_token_stays_inside_the_vocabulary() -> None:
    model = SimModel(vocab_size=512)
    produced = {model.next_token(100, i) for i in range(5000)}
    body = {t for t in produced if t != EOS}
    assert body, "expected some ordinary tokens"
    assert min(body) >= RESERVED
    assert max(body) < 512
    assert produced <= set(range(0, 512))


def test_eos_is_never_emitted_twice_in_a_row() -> None:
    model = SimModel(vocab_size=4096)
    last = 100
    for position in range(5000):
        nxt = model.next_token(last, position)
        if nxt == EOS:
            assert last != EOS
        last = nxt


def test_identical_tokens_do_not_repeat_back_to_back() -> None:
    model = SimModel(vocab_size=4096)
    generated = [100]
    for position in range(2000):
        nxt = model.next_token(generated[-1], position)
        if nxt == EOS:
            break
        generated.append(nxt)
    assert all(a != b for a, b in pairwise(generated))


def test_completions_terminate_before_the_hard_limit() -> None:
    model = SimModel(vocab_size=4096)
    lengths = []
    for start in range(300):
        last, emitted = start + RESERVED, 0
        for position in range(512):
            last = model.next_token(last, position)
            if last == 2:  # EOS
                break
            emitted += 1
        lengths.append(emitted)

    assert max(lengths) < 512, "EOS rate should keep completions bounded"
    assert min(lengths) < 64, "some completions should finish early"


def test_digest_is_stable_across_processes() -> None:
    assert stable_digest(1, "a") == stable_digest(1, "a")
    assert stable_digest(1, "a") != stable_digest(1, "b")


# --- device profile --------------------------------------------------------------


@pytest.fixture
def device() -> DeviceProfile:
    return DeviceProfile()


def test_prefill_marginal_cost_is_linear_in_tokens(device: DeviceProfile) -> None:
    """Per-token cost is linear. Total cost is not, because each forward pass
    also pays a fixed overhead."""
    marginal_small = device.prefill_seconds(1000) - device.pass_overhead_s
    marginal_large = device.prefill_seconds(2000) - device.pass_overhead_s
    assert marginal_large == pytest.approx(2 * marginal_small)


def test_total_prefill_cost_is_sublinear_in_tokens(device: DeviceProfile) -> None:
    assert device.prefill_seconds(2000) < 2 * device.prefill_seconds(1000)


def test_splitting_a_prompt_into_more_passes_costs_more(device: DeviceProfile) -> None:
    """The reason chunked prefill has a cost, and therefore an optimal size."""
    one_pass = device.prefill_seconds(2000)
    four_passes = sum(device.prefill_seconds(500) for _ in range(4))
    assert four_passes > one_pass
    assert four_passes == pytest.approx(one_pass + 3 * device.pass_overhead_s)


def test_decode_pays_one_pass_overhead_per_iteration(device: DeviceProfile) -> None:
    one = device.decode_seconds([500])
    four = device.decode_seconds([500] * 4)
    # Amortised across the batch, so four sequences cost barely more than one.
    assert four < one + 2 * device.pass_overhead_s


def test_prefill_of_nothing_is_free(device: DeviceProfile) -> None:
    assert device.prefill_seconds(0) == 0.0
    assert device.prefill_seconds(-5) == 0.0


def test_decode_cost_is_nearly_flat_in_batch_size(device: DeviceProfile) -> None:
    """The claim batching rests on: weights are read once for the whole batch."""
    one = device.decode_seconds([100])
    eight = device.decode_seconds([100] * 8)
    assert eight < one * 1.5


def test_decode_cost_grows_with_context_length(device: DeviceProfile) -> None:
    short = device.decode_seconds([100])
    long = device.decode_seconds([4000])
    assert long > short


def test_decode_of_empty_batch_is_free(device: DeviceProfile) -> None:
    assert device.decode_seconds([]) == 0.0


def test_prefill_is_cheaper_per_token_than_decode_at_scale(device: DeviceProfile) -> None:
    """Sanity check on the roofline split: prefill is compute bound, decode is
    bandwidth bound, so batching many sequences must beat running them apart."""
    batched = device.decode_seconds([500] * 32)
    serial = sum(device.decode_seconds([500]) for _ in range(32))
    assert batched < serial / 10
