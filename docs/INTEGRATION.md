# Integration guide

## Input contract

Capture pen-down to pen-up as one stroke and preserve point order. Send one mathematical expression at a time:

```json
{ "strokes": [{ "points": [{ "x": 12.5, "y": 8.0 }, { "x": 13.0, "y": 9.2 }] }] }
```

Coordinates can be pixels, canvas units, or another consistent 2D unit; v1 normalizes each expression. Do not resample with a different algorithm before calling the reference preprocessor. Keep stroke boundaries and the original ink so the user can correct an output.

## Python desktop or server

Use the included CPU example directly:

```bash
python examples/run_onnx.py --input request.json
```

For a service, keep one `onnxruntime.InferenceSession` per model/device rather than recreating it per request. Validate payload size and make recognition asynchronous or debounced; a fresh completion should supersede stale responses after the user keeps writing.

## Web and mobile

Use a pointer/stylus canvas that retains `{x, y}` samples per pointerdown/pointerup stroke. Run the ONNX graph with a platform-compatible ONNX Runtime package, then implement the preprocessing and EOS decoding from `examples/run_onnx.py` byte-for-byte. Batch size is always one. Prefer local inference; if a server is required, use an explicit user-visible privacy policy and consent flow.

For a note editor, render results as a pending suggestion. Offer accept, edit, reject, undo, and raw-ink restoration. Never discard the handwriting merely because a result has high model confidence.

## Rust native applications

`tract-onnx` is a suitable offline runtime. Load `model/handwriting-to-latex-v1.onnx`, form an `f32` tensor with shape `(1, point_count, 6)`, run it, then stop token decoding at the `<eos>` vocabulary ID. Verify the SHA-256 values in `model/manifest.json` before loading a downloaded artifact.

## PyTorch research use

`model/handwriting-to-latex-v1.pt` preserves the source checkpoint weights, vocabulary, configuration, epoch, and recorded validation history. It is intended for reproducibility and fine-tuning research. Use PyTorch's safe-loading options where applicable and rebuild the original encoder/attention/decoder architecture before loading it. Production clients should prefer the portable ONNX artifact.

## Output handling

The output `confidence` is an average selected-token probability. It is useful for UI triage but is not a calibrated probability that the LaTeX is correct. Render LaTeX in a sandboxed/math-safe renderer, surface parse errors, and give users an editing path.
