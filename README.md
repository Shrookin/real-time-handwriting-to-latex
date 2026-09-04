# Real-Time Handwriting to LaTeX v1

An offline model that turns **online handwritten mathematical ink**—the pen strokes captured while someone writes—into LaTeX. It is intended for real-time note-taking, whiteboards, stylus apps, accessibility tooling, and mathematical-input experiments.

> Released under [Apache-2.0](LICENSE). See [attribution and release basis](docs/ATTRIBUTION_AND_LICENSE.md).

## What it does

- Accepts one online-ink mathematical expression as ordered strokes of `{x, y}` points.
- Produces a tokenized LaTeX string with a per-token confidence-derived score.
- Runs fully offline using the included ONNX model on CPU.
- Covers common operators, fractions, roots, matrices, integrals, Greek letters, scripts, and `cases` layout.

It is **not** an image/OCR model. A photograph, scan, raster canvas, or PDF must first be converted to ordered pen strokes by a separate recognizer. It also does not segment a full page into expressions; pass one expression at a time.

## Quick start: ONNX Runtime

```bash
git clone https://github.com/Shrookin/real-time-handwriting-to-latex.git
cd real-time-handwriting-to-latex
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/run_onnx.py --input examples/sample_request.json
```

The program reads the online-ink JSON shown below and writes a JSON response containing `latex`, `confidence`, and the decoded tokens.

```json
{
  "strokes": [
    { "points": [{ "x": 10, "y": 20 }, { "x": 20, "y": 20 }] },
    { "points": [{ "x": 15, "y": 10 }, { "x": 15, "y": 30 }] }
  ]
}
```

For browser, mobile, desktop, Rust, and server integration patterns, see [Integration guide](docs/INTEGRATION.md). For model limits and measured performance, see the [model card](docs/MODEL_CARD.md).

## Included artifacts

| File | Purpose | SHA-256 |
| --- | --- | --- |
| `model/handwriting-to-latex-v1.onnx` | CPU-friendly ONNX greedy decoder | `33832e523d1d60e6bdcf8b0482d31d4a3c416c737532b95ac4376e07370be16a` |
| `model/handwriting-to-latex-v1-vocab.json` | Decoder vocabulary and EOS token | `e2a77b00b6e4317148ee3cf6267a979175bae1a49f25076a62753ee35d70bb9b` |
| `model/handwriting-to-latex-v1.pt` | Original PyTorch checkpoint for research provenance | `0106db8171a25e4b6e1dfdfc41c4292c836bdd22c627465c02b3d982672da70f` |
| `model/manifest.json` | Version, input contract, hashes, and provenance | — |

Verify a downloaded model before deploying it:

```bash
sha256sum model/handwriting-to-latex-v1.onnx
# PowerShell: Get-FileHash .\\model\\handwriting-to-latex-v1.onnx -Algorithm SHA256
```

## Training and evaluation

The model is a bidirectional GRU encoder with additive attention and a greedy LaTeX-token decoder. It has a 192-dimensional hidden state, a 538-token vocabulary, six point features, a 384-point input cap, and a 64-token output cap.

The selected checkpoint was trained on a staged mixture of:

- MathWriting online handwritten math expressions;
- the 2011–2013 CROHME expression-training split; and
- a 100,000-example synthetic piecewise-layout corpus made by recombining MathWriting glyph strokes. It adds structural variation; it is not a new-writer dataset.

Its final training stage used 329,332 MathWriting replay records, 11,094 CROHME records, and 100,000 synthetic piecewise records. On the held-out validations for that stage, it achieved 46.36% exact match on 15,674 MathWriting expressions (14.82% mean token error), 21.04% exact match on 984 CROHME expressions, and 94.54% exact match on 10,000 synthetic piecewise examples. These scores are not interchangeable: the last score measures structural generalization from recombined glyphs, not unfamiliar handwriting.

Read [how v1 was trained](docs/TRAINING.md), [benchmark interpretation](docs/BENCHMARKS.md), and the [roadmap](docs/ROADMAP.md) before using the model in a product or comparing it to image-to-LaTeX systems.

## Contributing and future releases

Please open an issue with a minimal **synthetic or consented** stroke sample, target LaTeX, predicted LaTeX, and runtime/version. Do not upload private notes or unconsented handwriting. The future data and quality plan is in [ROADMAP.md](docs/ROADMAP.md).

## Citation and attribution

If you use the model, cite this repository and preserve the source attributions in [ATTRIBUTION_AND_LICENSE.md](docs/ATTRIBUTION_AND_LICENSE.md). The raw training datasets are not included here.
