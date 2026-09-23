#!/usr/bin/env python3
"""Generate known subpixel motion, recover it via the public CLI, and measure it.

Run from any directory: python experiments/known_motion.py --output-dir /tmp/motion-demo
Only the project's CPU requirements are needed. No camera, audio, or network access.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.io.wavfile import read

FPS = 1000
FRAMES = 1000
SIZE = 96
TONES = (80, 200)
AMPLITUDES = (0.2, 0.1)


def make_video(path, moving):
    # Save at playback FPS=30 deliberately: the CLI must honor --fps 1000.
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), 30, (SIZE, SIZE))
    if not writer.isOpened():
        raise RuntimeError('OpenCV cannot write FFV1 AVI; a lossless codec is required for this experiment')
    x, y = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
    try:
        for n in range(FRAMES):
            dx = sum(a * np.sin(2 * np.pi * f * n / FPS) for f, a in zip(TONES, AMPLITUDES)) if moving else 0
            # Analytic texture shifted before 8-bit quantization. This avoids
            # interpolation or per-frame random noise becoming the test signal.
            gray = np.rint(128 + 40 * np.cos(2 * np.pi * (x - dx) / 12)
                           + 30 * np.cos(2 * np.pi * ((x - dx) / 19 + y / 17))).astype('uint8')
            writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()


def measure(path):
    rate, raw = read(path)
    if rate != FPS or len(raw) != FRAMES or raw.dtype != np.int16 or raw.ndim != 1:
        raise AssertionError(f'Incorrect WAV format/timing: rate={rate}, shape={raw.shape}, dtype={raw.dtype}')
    # Exclude 100 ms at either edge for the zero-phase filter's edge transient.
    samples = raw[100:-100].astype(float)
    t = np.arange(len(samples)) / rate
    design = np.column_stack([np.sin(2 * np.pi * f * t) for f in TONES]
                             + [np.cos(2 * np.pi * f * t) for f in TONES]
                             + [np.ones(len(t))])
    fit = np.linalg.lstsq(design, samples, rcond=None)[0]
    amplitudes = np.hypot(fit[:2], fit[2:4])
    spectrum = abs(np.fft.rfft(samples - np.mean(samples)))
    return {
        'sample_rate_hz': rate,
        'sample_count': len(raw),
        'duration_seconds': len(raw) / rate,
        'nonzero_samples': int(np.count_nonzero(raw)),
        'dominant_frequency_hz': float(np.fft.rfftfreq(len(samples), 1 / rate)[np.argmax(spectrum)]) if samples.any() else None,
        'tone_amplitudes_pcm': amplitudes.tolist(),
        'ratio_200_to_80': float(amplitudes[1] / amplitudes[0]) if amplitudes[0] else None,
        'two_tone_r_squared': float(1 - np.var(samples - design @ fit) / np.var(samples)) if np.var(samples) else None,
        'wav_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def run(output_dir, source_root):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ('motion', 'static'):
        make_video(output_dir / f'{name}.avi', moving=name == 'motion')
    results = {}
    for name, video, flags in [('unfiltered', 'motion', []), ('bandpass', 'motion', ['-fl', '50', '-fh', '120']), ('static', 'static', [])]:
        command = [sys.executable, str(source_root / 'motionized_audio.py'), '-i', str(output_dir / f'{video}.avi'),
                   '-o', str(output_dir / f'{name}.wav'), '--fps', str(FPS), *flags]
        completed = subprocess.run(command, capture_output=True, text=True)
        (output_dir / f'{name}.log').write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f'{name} failed; see {output_dir / (name + ".log")}')
        results[name] = measure(output_dir / f'{name}.wav')
    rejection_db = float(20 * np.log10(results['unfiltered']['ratio_200_to_80'] / results['bandpass']['ratio_200_to_80']))
    checks = {
        'dominant_frequency_within_one_fft_bin': abs(results['unfiltered']['dominant_frequency_hz'] - TONES[0]) <= 1.25,
        'known_tones_explain_at_least_99_percent_variance': results['unfiltered']['two_tone_r_squared'] >= 0.99,
        'amplitude_ratio_within_0_03_of_expected_0_5': abs(results['unfiltered']['ratio_200_to_80'] - 0.5) <= 0.03,
        'bandpass_relative_rejection_at_least_35_db': rejection_db >= 35,
        'bandpass_retains_80_hz': abs(results['bandpass']['dominant_frequency_hz'] - TONES[0]) <= 1.25,
        'static_control_is_exactly_silent': results['static']['nonzero_samples'] == 0,
    }
    report = {
        'experiment': 'analytic-horizontal-two-tone-subpixel-motion-v1',
        'environment': {'python': platform.python_version(), 'platform': platform.system() + '-' + platform.machine(),
                        **{name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'opencv-python', 'dtcwt')}},
        'input': {'capture_fps': FPS, 'container_fps': 30, 'frames': FRAMES, 'size': [SIZE, SIZE],
                  'frequencies_hz': TONES, 'displacement_amplitudes_pixels': AMPLITUDES, 'codec': 'FFV1', 'analysis_edge_exclusion_samples': 100},
        'results': results,
        'bandpass_relative_rejection_db': rejection_db,
        'checks': checks,
        'passed': all(checks.values()),
    }
    (output_dir / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='Directory for generated AVI, recovered WAV, logs, and results.json')
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1], help='Source checkout to evaluate (defaults to this checkout)')
    args = parser.parse_args()
    sys.exit(0 if run(args.output_dir.resolve(), args.source_root.resolve()) else 1)
