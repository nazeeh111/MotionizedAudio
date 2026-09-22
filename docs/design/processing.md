# MotionizedAudio processing

The command entry point delegates to the compatible `visualmic` module. Both
commands share argument validation, defaults, frame reading, reconstruction,
and WAV output. Branding does not select a different algorithm.

## Pipeline

1. Read video frames, convert to grayscale, and optionally crop the selected region.
2. Decompose each frame with a two-dimensional dual-tree complex wavelet transform.
3. Compare complex coefficient phases against the reference frame and weight the
   resulting phase changes by squared amplitude.
4. Optionally apply the temporal frequency filter, align the orientation/scale
   signals by cross-correlation, sum them, and normalize the waveform.
5. Write signed 16-bit PCM audio at the capture-rate override or video frame rate.

The CPU implementation processes one frame at a time. The CUDA implementation
batches transforms on the GPU and returns phase signals for CPU postprocessing.
CPU and GPU results can differ numerically because their transform libraries and
floating-point precision differ.

## Runtime compatibility

The CPU transform uses NumPy interfaces removed in version 2. The requirements
therefore constrain NumPy below 2 and use a compatible OpenCV release. The
recommended local Python versions are 3.11 and 3.12. GPU installation is supplied
through the CUDA container build.

## Verification

The existing test suite exercises signal shifts, WAV output, temporal processing,
synthetic frame extraction, parameter selection, and command validation. The
renamed command was also compared with the retained baseline using five video
processing configurations. See the root VERIFICATION.md for evidence and limits.

Physical sound recovery depends on the recording: capture frequency, motion
amplitude, texture, lighting, and camera stability. Synthetic equivalence does
not establish recovery quality for every scene.
