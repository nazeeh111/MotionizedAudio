"""Regression cases found by the deterministic-video audit."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import visualmic

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args):
    return subprocess.run([sys.executable, str(ROOT / 'motionized_audio.py'), *args], capture_output=True, text=True)


@pytest.mark.parametrize('frames', [12, 16, 27])
def test_short_bandpass_is_explicitly_rejected(frames):
    phases = np.sin(2 * np.pi * 80 * np.arange(frames) / 1000).reshape(-1, 1, 1)
    with pytest.raises(ValueError, match='at least 28 frames'):
        visualmic.postprocess_phase_signals(phases, frames, 1, 1, 0, 0, 1000, 50, 120)


@pytest.mark.parametrize('flag,value', [('--fps', 'nan'), ('--fps', 'inf'), ('--freq-low', '-1'), ('--freq-high', 'nan'), ('--batch-size', '0')])
def test_invalid_numbers_rejected_before_opening_input(flag, value):
    result = run_cli('-i', 'does-not-exist.avi', flag, value)
    assert result.returncode != 0
    assert flag in result.stderr + result.stdout
    assert 'Traceback' not in result.stderr


@pytest.mark.parametrize('low,high', [(500, None), (499, 501), (None, 501)])
def test_nyquist_violation_is_not_silently_skipped_or_clamped(low, high):
    phases = np.zeros((64, 1, 1))
    with pytest.raises(ValueError, match='Nyquist'):
        visualmic.postprocess_phase_signals(phases, 64, 1, 1, 0, 0, 1000, low, high)


@pytest.fixture
def short_video(tmp_path):
    path = tmp_path / 'clip.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), 1000, (32, 32))
    assert writer.isOpened()
    for n in range(16):
        frame = np.full((32, 32, 3), n, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_input_video_cannot_be_overwritten(short_video):
    before = short_video.read_bytes()
    result = run_cli('-i', str(short_video), '-o', str(short_video))
    assert result.returncode != 0
    assert 'input' in result.stderr.lower()
    assert short_video.read_bytes() == before


def test_short_filter_cli_preserves_previous_output(short_video, tmp_path):
    output = tmp_path / 'prior.wav'
    output.write_bytes(b'previous output')
    result = run_cli('-i', str(short_video), '-o', str(output), '-fl', '50', '-fh', '120')
    assert result.returncode != 0
    assert 'at least 28 frames' in result.stderr
    assert 'Traceback' not in result.stderr
    assert output.read_bytes() == b'previous output'


def test_fractional_capture_rate_preserves_timing_and_pitch(tmp_path):
    from scipy.io.wavfile import read
    capture_fps = 29.97
    t = np.arange(2997) / capture_fps
    samples = np.sin(2 * np.pi * 3 * t)
    path = tmp_path / 'fractional.wav'
    visualmic.save_wav(samples, str(path), capture_fps)
    rate, audio = read(path)
    assert rate == 30
    assert abs(len(audio) / rate - 100) <= 1 / rate
    frequencies = np.fft.rfftfreq(len(audio), 1 / rate)
    assert abs(frequencies[np.argmax(abs(np.fft.rfft(audio))) ] - 3) <= 0.01


def test_cli_does_not_truncate_fractional_fps(short_video, tmp_path):
    from scipy.io.wavfile import read
    output = tmp_path / 'fractional-cli.wav'
    result = run_cli('-i', str(short_video), '-o', str(output), '--fps', '29.97')
    assert result.returncode == 0, result.stderr
    rate, data = read(output)
    assert rate == 30
    assert abs(len(data) / rate - 16 / 29.97) <= 1 / rate


def test_unrepresentable_wav_rate_preserves_previous_output(short_video, tmp_path):
    output = tmp_path / 'prior.wav'
    output.write_bytes(b'previous output')
    result = run_cli('-i', str(short_video), '-o', str(output), '--fps', '2147483648')
    assert result.returncode != 0
    assert 'Traceback' not in result.stderr
    assert 'sample rate' in result.stderr.lower()
    assert output.read_bytes() == b'previous output'


def test_failed_wav_write_preserves_previous_output(tmp_path, monkeypatch):
    output = tmp_path / 'prior.wav'
    output.write_bytes(b'previous output')

    def interrupted_write(destination, rate, samples):
        # Exercise the real file boundary: a writer can fail after partial output
        # (for example when the disk fills). Only the failing encoder is replaced.
        with open(destination, 'wb') as stream:
            stream.write(b'incomplete WAV')
        raise OSError('simulated disk full after partial write')

    monkeypatch.setattr(visualmic, 'write', interrupted_write)
    with pytest.raises(OSError, match='disk full'):
        visualmic.save_wav(np.zeros(32), str(output), 1000)
    assert output.read_bytes() == b'previous output'
    assert list(tmp_path.iterdir()) == [output]
