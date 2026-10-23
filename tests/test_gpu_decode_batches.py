"""GPU batching control flow with a NumPy adapter; no CUDA transform is run."""
import sys
import types

import numpy as np
import pytest

import visualmic

REAL_POSTPROCESS = visualmic.postprocess_phase_signals


class Tensor:
    """Only the tensor operations used by extract_audio_gpu's batch loop."""
    def __init__(self, values):
        self.values = np.asarray(values)

    def to(self, device):
        return self

    def clone(self):
        return Tensor(self.values.copy())

    def __getitem__(self, key):
        return Tensor(self.values[key])

    def __mul__(self, other):
        return Tensor(self.values * other.values)

    def __add__(self, other):
        return Tensor(self.values + other.values)

    def __sub__(self, other):
        return Tensor(self.values - other.values)

    def sum(self, dim):
        return Tensor(self.values.sum(axis=dim))

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class Capture:
    def __init__(self, decoded, none_frame=False):
        self.decoded = decoded
        self.reads = 0
        self.released = False
        self.none_frame = none_frame

    def read(self):
        index = self.reads
        self.reads += 1
        if index >= self.decoded:
            return (True, None) if self.none_frame else (False, None)
        return True, np.full((8, 8, 3), index, dtype=np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def adapter(monkeypatch):
    batches = []
    outputs = []
    failure = []

    class Transform:
        def __init__(self, J, **kwargs):
            self.levels = J

        def to(self, device):
            return self

        def __call__(self, tensor):
            if failure:
                raise RuntimeError(failure[0])
            indexes = tensor.values[:, 0, 0, 0]
            batches.append(indexes.tolist())
            values = np.zeros((len(indexes), 1, 6, 1, 1, 2))
            values[..., 0] = 1.0
            values[..., 1] = indexes[:, None, None, None, None]
            return None, [Tensor(values.copy()) for _ in range(self.levels)]

    torch = types.SimpleNamespace(
        device=lambda name: name,
        from_numpy=Tensor,
        atan2=lambda a, b: Tensor(np.arctan2(a.values, b.values)),
        cuda=types.SimpleNamespace(get_device_name=lambda index: 'stub, no CUDA'),
    )
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'pytorch_wavelets', types.SimpleNamespace(DTCWTForward=Transform))

    def postprocess(phases, count, *args):
        outputs.append((phases.copy(), count))
        return phases

    monkeypatch.setattr(visualmic, 'postprocess_phase_signals', postprocess)
    return batches, outputs, failure


def extract(cap, advertised, ref_index=0):
    return visualmic.extract_audio_gpu(
        cap, advertised, nlevels=1, n_orient=6, ref_index=ref_index,
        ref_orient=0, ref_level=0, fps=1000, batch_size=4,
    )


@pytest.mark.parametrize('decoded,advertised,expected', [
    (3, 6, [[0, 1, 2]]),
    (7, 10, [[0, 1, 2, 3], [4, 5, 6]]),
    (6, 6, [[0, 1, 2, 3], [4, 5]]),
    (8, 8, [[0, 1, 2, 3], [4, 5, 6, 7]]),
])
def test_all_decoded_frames_are_processed_once(adapter, decoded, advertised, expected):
    batches, outputs, _ = adapter
    cap = Capture(decoded)
    phases = extract(cap, advertised)
    assert cap.released
    assert batches == expected
    assert outputs[0][1] == decoded
    assert phases.shape == (decoded, 1, 6)
    # With stub reference 1+0i, each sample's weighted phase is independent.
    indexes = np.arange(decoded, dtype=float)
    np.testing.assert_allclose(phases[:, 0, 0], (1 + indexes**2) * np.arctan2(indexes, 1.0))


def test_true_decode_flag_with_none_frame_flushes_pending(adapter):
    cap = Capture(3, none_frame=True)
    assert extract(cap, 6).shape[0] == 3
    assert adapter[0] == [[0, 1, 2]]
    assert cap.released


def test_reference_in_pending_batch_keeps_original_frame_index(adapter):
    cap = Capture(7)
    phases = extract(cap, 10, ref_index=5)
    np.testing.assert_array_equal(phases[:6], 0.0)
    expected = (1 + 6.0**2) * np.arctan2(6.0 - 5.0, 1 + 6.0 * 5.0)
    np.testing.assert_allclose(phases[6], expected)
    assert adapter[1][0][1] == 7


def test_reference_not_decoded_is_not_taken_from_failed_read(adapter):
    cap = Capture(7)
    phases = extract(cap, 10, ref_index=7)
    np.testing.assert_array_equal(phases, np.zeros((7, 1, 6)))
    assert cap.released


def test_zero_decoded_frames_still_rejected(adapter):
    cap = Capture(0)
    with pytest.raises(SystemExit) as error:
        extract(cap, 6)
    assert error.value.code == 1
    assert cap.released
    assert adapter[0] == []
    assert adapter[1] == []


def test_out_of_memory_keeps_existing_exit_and_release(adapter):
    adapter[2].append('out of memory')
    cap = Capture(3)
    with pytest.raises(SystemExit) as error:
        extract(cap, 3)
    assert error.value.code == 1
    assert cap.released
    assert adapter[1] == []


def test_other_transform_runtime_errors_still_propagate(adapter):
    adapter[2].append('transform failed')
    with pytest.raises(RuntimeError, match='transform failed'):
        extract(Capture(3), 3)


def test_early_stop_still_checks_filter_against_actual_frame_count(adapter, monkeypatch):
    monkeypatch.setattr(visualmic, 'postprocess_phase_signals', REAL_POSTPROCESS)
    cap = Capture(3)
    with pytest.raises(ValueError, match='at least 28 frames; got 3'):
        visualmic.extract_audio_gpu(
            cap, 64, nlevels=1, n_orient=6, ref_index=0,
            ref_orient=0, ref_level=0, fps=1000, batch_size=4,
            freq_low=50, freq_high=120,
        )
    assert cap.released
