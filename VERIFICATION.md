# Verification

## End-to-end recovery and correctness audit (2026-09-23)

A fresh Python 3.12.13 environment installed the declared CPU and development
requirements from the available local package cache: NumPy 1.26.4, SciPy 1.17.1,
OpenCV 4.11.0.86, dtcwt 0.14.0, pytest 8.4.2 and Ruff 0.16.8. No system packages
or prior virtual environment were inherited. The source was exported without
Git metadata or caches before rerunning the experiment.

- `python -m pytest tests -q`: **45 passed, 7 CUDA tests skipped**.
- `ruff check .` and `git diff --check`: passed.
- `python experiments/known_motion.py --output-dir /tmp/motionized-audio-demo`:
  six scientific acceptance checks passed. Every WAV has the expected format,
  sample count and one-second timing. See the [method](experiments/README.md)
  and [machine-readable measurements](docs/known-motion-results.json).
- Same experiment run against archived baseline commit
  `df75677a0fcc95fca69d39947a9f99741215cbb6`: all three WAVs are byte-identical
  to the revised source for valid integer-rate input.

The audit reproduced these failures before changing the implementation:

| Failure | Revised behavior |
| --- | --- |
| Bandpass at 12 frames silently bypassed; 16 and 27 frames raised a SciPy padding traceback | Explicit minimum of 28 frames for bandpass, 16 for a single cutoff; actionable error, previous output preserved |
| Out-of-Nyquist cutoffs were silently skipped or clamped | Reject invalid requested frequency ranges and explain the capture-rate limit |
| Nonfinite FPS/cutoffs, negative cutoffs and zero batch size escaped early validation | Reject invalid arguments before opening input |
| Output equal to input overwrote the original video | Reject identical paths, symlinks and existing hard-link aliases |
| 29.97 fps was written as 29 Hz, changing playback duration and pitch | Resample fractional capture rates to the nearest positive integer WAV rate; integer-rate results remain unchanged |
| A finite but unrepresentable FPS of 2147483648 caused a WAV-header traceback and truncated existing output to zero bytes | Validate the 16-bit mono WAV sample/byte-rate bounds; reject before opening output |
| An encoder or disk error after a partial write destroyed existing output | Stage beside the output and publish with atomic replacement only after successful encoding; clean partial temporary files |

Independent review reproduced the oversized-rate truncation against both the prior committed version and the proposed changes. New regressions failed on the old writer and pass with atomic output, including a simulated disk error after an actual partial file write. The full suite and known-motion experiment were rerun after this repair.

Regression evidence includes a 100-second, 3 Hz signal sampled at 29.97 Hz,
which now produces 3,000 samples at 30 Hz with a 3 Hz dominant peak. Resampling
uses a bounded rational approximation and rounds duration to an output sample;
it cannot improve the original temporal resolution. Resampling can overshoot,
so PCM values are clipped before integer conversion instead of wrapping.

CUDA processing and Docker execution remain unverified locally. CI now runs
the known-motion experiment in the CPU image; adding the job is not evidence
that a remote CI run passed. The fixture is ideal image displacement, not real
acoustic speech recovery or calibrated pressure measurement.

# Earlier equivalence checks

Verified locally on macOS ARM64 with Python 3.12.13, NumPy 1.26.4,
SciPy 1.17.1, OpenCV 4.11.0 and dtcwt 0.13.0.

- `python -m pytest tests/ -q`: 28 passed, 7 CUDA tests skipped.
- `ruff check .`: passed with the explicit upstream lint scope (E4/E7/E9/F).
- `python motionized_audio.py --help` and `--version`: passed.
- `git diff --check`: passed.
- Compared the upstream `visualmic.py` at commit
  `39e90eb1a40d2e21e69e5e3af200a4a8673c897c` against the renamed CLI using
  a deterministic 32-frame, 128×128 MJPEG synthetic video. WAV files were
  byte-identical for five configurations: defaults, 2200 Hz override,
  80–900 Hz bandpass at 2200 Hz, 64×64 region of interest, and two wavelet
  levels with near_sym_a/qshift_a filters.
- Missing input, missing file, and malformed region-of-interest arguments
  returned matching nonzero exit codes in both entry points.
- Source comparison confirmed that visualmic.py differs from upstream only
  in its module description and CLI description. Numerical processing is unchanged.

## Dependency correction

The original requirements allowed NumPy 2, but dtcwt 0.13 calls `np.asfarray`,
which NumPy 2 removed. The failure was reproduced with the original source.
The CPU requirements now constrain NumPy below 2 and OpenCV below 4.12 to
retain compatible wheels. Both failing extraction tests pass with these bounds.

## Limits

These are synthetic-video equivalence checks, not a replication of the MIT
speech-recovery experiments. The large external MIT video dataset was not
downloaded. CUDA processing was not executed because no NVIDIA GPU is
available on this Mac. Docker builds were not run locally. CI retains the
CPU/GPU container build checks and now also runs the CPU regression tests.
The restyled illustration is explanatory artwork, not experimental evidence.

## Visual refresh verification

The illustration retains the light scientific-pipeline layout, replaces the cup with an unbranded silver can, and uses indigo accents. It is explanatory artwork, not evidence of a new can experiment. The numerical source and CLI code were unchanged. After this update, the CPU suite again passed 28 tests, with 7 CUDA tests skipped; Ruff passed.

The follow-up flowchart refinement adds numbered stages, consistent arrows, subtle rounded panels, and an explicit illustrative-workflow footer. Recovered output is labeled relative rather than calibrated pressure. This follow-up changes only artwork and this verification note; the previously tested numerical code is unchanged.
