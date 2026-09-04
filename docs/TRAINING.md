# How v1 was trained

## Release identity and lineage

The public release is **v1**. Its internal provenance name is `expression-torch-v4-real-structure-replay-candidate`; the different names distinguish the public product release from the many prior NewNotes experiments.

The final checkpoint was initialized from a v3 piecewise-replay checkpoint, which was initialized from a v3 curriculum/balanced checkpoint. It is therefore a staged replay model, not a single from-scratch run. The final v4 stage ran for one epoch on an NVIDIA GeForce RTX 4060 Laptop GPU using PyTorch 2.11.0+cu130, CUDA 13.0, a batch size of 32, learning rate `2e-6`, label smoothing `0.05`, no mixed precision, and cuDNN disabled for that run.

## Data used in the final stage

| Source | Final-stage records | Split/use | Notes |
| --- | ---: | --- | --- |
| MathWriting | 329,332 | Training replay; separate official valid split for evaluation | Online handwritten math in InkML |
| CROHME 2011–2013 | 11,094 | Expression training replay | CROHME 2014 expression ground truth held out for evaluation |
| `piecewise-curated-v2` | 100,000 | Synthetic training replay | Recombined MathWriting glyph strokes with varied cases layouts |

Record counts are training-stage cache counts, not a claim that all records are unique writers or independent expressions. The synthetic piecewise corpus deliberately adds rows, conditions, relation operators, spacing, baseline drift, scale, jitter, and eight geometry profiles. It does not add new writers.

The artifact contains **zero CROHME23 training samples**. CROHME23 was investigated in separate experiments but is not a dependency of v1.

## Training pipeline

1. Parse online-ink InkML and normalize labels with the MathWriting token convention.
2. Convert every expression into at most 384 sampled points. Preserve the first point of each stroke; normalize coordinates and deltas by expression extent.
3. Cache fixed-shape NumPy shards to train without retaining the full corpus in memory.
4. Initialize the bidirectional-GRU/attention model from the prior checkpoint and copy compatible vocabulary rows.
5. Train with replay: eight MathWriting batches per CROHME batch and eight MathWriting batches per piecewise batch.
6. Greedily decode held-out formulas, normalize operator/script output, and record exact-match and token-error metrics.
7. Export the frozen checkpoint to an ONNX graph that performs the same fixed-length greedy decode. The application stops at the vocabulary's EOS token.

The final run processed 12,864 batches: 1,286 CROHME replay batches, 1,286 piecewise replay batches, and no CROHME23 replay batches. The checkpoint includes a 192-unit hidden state, dropout 0.1, 64-token decoder cap, six input features, and 538 vocabulary tokens.

## Reproducing a comparable training run

The source package used for cache construction, training, evaluation, and synthetic corpus generation is included under [`training/`](../training/README.md). Raw datasets and generated caches are intentionally not distributed in this repository. Obtain each dataset directly from its source, comply with its terms, preserve its original split boundaries, and use the exact preprocessing described above. A comparable mixed run needs:

```text
python -m training.recognition_service.expression_mixed_train \
  --mathwriting-cache-dir <mathwriting-replay-cache> \
  --crohme-cache-dir <crohme-cache> \
  --piecewise-cache-dir <piecewise-cache> \
  --init-checkpoint <v3-parent.pt> \
  --output <v1-derived.pt> \
  --mathwriting-limit 0 --mathwriting-per-crohme 8 \
  --piecewise-per-mathwriting 8 --epochs 1 --batch-size 32 \
  --learning-rate 0.000002 --label-smoothing 0.05 \
  --length-bucket-size 16 --device cuda --no-amp --disable-cudnn
```

This command documents the final-stage settings; it is not enough on its own to recreate the identical artifact because it depends on the frozen parent checkpoint, cache ordering, seed handling, and source snapshots. Any new model must be evaluated on untouched MathWriting, CROHME, and separately labeled structure data before promotion.
