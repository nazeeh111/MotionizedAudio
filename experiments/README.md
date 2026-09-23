# Known-motion recovery experiment

This is an end-to-end CPU check with known physical displacement, not a speech
recovery demonstration. It generates video, runs the public command in a separate
process, reads the saved PCM WAV, and fails with a nonzero exit code if measured
results violate the declared acceptance criteria. No downloaded data is needed.

From a fresh checkout with Python 3.11 or 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/known_motion.py --output-dir /tmp/motionized-audio-demo
```

The output directory contains `motion.avi`, `static.avi`, three recovered WAVs,
CLI logs, and `results.json`. Use a new directory to retain earlier results.
`--source-root /path/to/another/checkout` evaluates a baseline with the same
experiment. Outputs are disposable generated data. The WAVs use a 1000 Hz sample
rate to match capture timing; some audio players may require external resampling.

## Signal and assumptions

The fixture is a 96×96 analytic grayscale texture translated horizontally by

```
dx[n] = 0.2 sin(2π 80 n / 1000) + 0.1 sin(2π 200 n / 1000) pixels.
```

Spatial cosine gratings at two angles provide textured edges. Each image is
sampled directly from the translated analytic function, quantized to 8 bits and
encoded losslessly as FFV1. There is no interpolation between raster frames and
no random motion. All 1,000 frames represent one second at a 1,000 Hz capture
rate. The container deliberately reports 30 fps; `--fps 1000` must override it.
The static control renders the same texture at zero displacement.

The recovery uses the default three-level complex wavelet transform on CPU.
A second run requests a 50–120 Hz bandpass. Analysis excludes 100 samples at
each end to reduce filter edge transients. Least-squares sine and cosine fits
at the two known frequencies measure phase-independent amplitudes. The ratio
of 200 Hz to 80 Hz is compared within each output because each WAV is separately
normalized. Absolute PCM amplitude across files is not an attenuation measure.

Acceptance criteria, fixed in the script:

- Output is mono signed 16-bit PCM, with 1,000 samples at 1,000 Hz.
- The largest unfiltered Fourier component is within one 1.25 Hz analysis bin of 80 Hz.
- The two prescribed tones explain at least 99% of unfiltered signal variance.
- The unfiltered amplitude ratio is within 0.03 of the expected 0.5.
- Bandpass filtering retains the 80 Hz peak and improves the 200-to-80 Hz ratio by at least 35 dB.
- The complete static-control WAV contains only zeros.

## Measured result

[Recorded local results](../docs/known-motion-results.json) show an 80 Hz peak,
0.498617 amplitude ratio, 0.9999916 two-tone R², and 69.04 dB relative rejection.
The static output is exactly silent. All three WAVs have the expected one-second
duration. The prior committed version and revised version produce identical
WAV bytes on these valid integer-rate inputs; the fixes affect invalid input,
short-filter handling, input preservation, and fractional-rate output timing.
The recorded environment uses dtcwt 0.14.0, which the current requirements allow.
Thresholds allow numerical differences across supported CPU environments;
the recorded SHA-256 hashes are local evidence, not cross-platform guarantees.

## Limits

This verifies recovery of ideal image displacement. It does not simulate a
material's acoustic transfer function, rolling shutter, motion blur, illumination
flicker, camera noise, compression artifacts, large motion, or speech. The phase
signals are normalized relative values, not calibrated pressure or displacement.
A noise-free static control is weaker evidence than a real stationary camera
recording. Temporal content above half the actual capture rate aliases and
cannot be recovered correctly by this experiment. No GPU or microphone was used.
