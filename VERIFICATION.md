# Verification

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
