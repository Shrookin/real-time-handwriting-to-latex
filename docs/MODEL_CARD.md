# Model card: Real-Time Handwriting to LaTeX v1

## Intended use

v1 converts a single, ordered sequence of stylus or pointer strokes into LaTeX while the user writes. It is designed for mathematical note apps, whiteboards, accessibility input, and prototyping. It can run locally without sending handwriting to a server.

The input is online ink, not an image: each stroke contains ordered `{x, y}` points. The model does not detect formula regions in a page, recognize scanned handwriting, validate mathematical correctness, or guarantee compilable LaTeX.

## Architecture and interface

The model is a bidirectional GRU encoder (six input features, 192 hidden units), additive attention, and an autoregressive GRU decoder. It predicts from a 538-token LaTeX vocabulary by greedy decoding. The exported ONNX graph accepts `float32[1, point_count, 6]` and returns 64 token IDs plus their selected-token probabilities.

Each expression is normalized independently. The six features are normalized x/y coordinates, normalized x/y deltas, `new_stroke`, and `same_stroke`. Inputs longer than 384 points are deterministically subsampled while keeping every stroke's first point. Decode stops at `<eos>` or after 64 positions.

## Quality measurements

| Validation set | Samples | Exact match | Mean token error | What it measures |
| --- | ---: | ---: | ---: | --- |
| MathWriting valid | 15,674 | 46.36% | 14.82% | Broad held-out online-ink formula recognition |
| CROHME 2014 expression validation | 984 | 21.04% | 53.75% | Transfer to a distinct historical writer/source distribution |
| Synthetic piecewise validation | 10,000 | 94.54% | 0.24% | Structural cases-layout generalization from recombined source glyphs |

Exact match is strict: the complete normalized token sequence must match. The piecewise score must not be read as an unfamiliar-writer score.

## Known limitations

- Superscripts/subscripts, long expressions, and non-left-to-right writing order are important remaining failure modes.
- Confidence is the mean probability of the selected decoded tokens. It is not calibrated correctness probability and should not silently accept a result.
- Results depend on stroke order, grouping, and coordinate geometry. Callers should retain the original ink and offer edit/correct/retry actions.
- This is a research-quality single-expression model, not a general OCR or document-understanding system.
- The model's vocabulary is fixed. Unsupported notation may be emitted as incorrect known tokens or `<unk>`.

## Privacy and safety

Run locally where possible. If an application collects corrections, make collection opt-in, minimize the stroke data, and do not upload note context by default. Do not use v1 as the sole mechanism for grading, accessibility decisions, or high-consequence mathematical work.
