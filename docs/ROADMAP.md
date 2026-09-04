# Roadmap

## Next quality work

1. Add a genuinely writer-held-out benchmark for cases, aligned layouts, and long expressions; do not use synthetic recombination as a substitute.
2. Improve superscript/subscript and non-left-to-right continuation handling with measured, held-out stroke-order evaluations.
3. Add constrained decoding or validity-aware post-processing without hiding an uncertain result.
4. Calibrate confidence and expose alternatives so applications can ask users to confirm low-confidence output.
5. Add an evaluation corpus with consented correction data, strict privacy minimization, documented provenance, and clear deletion/opt-out procedures.

## Runtime and developer experience

1. Publish tested Web, Rust, Kotlin/Android, Swift/iPadOS, and server examples that use the same preprocessing contract.
2. Add model integrity verification and semantic versioning to every downloadable release.
3. Benchmark ONNX Runtime and native accelerators on common tablets and laptops.
4. Release a smaller model only if it preserves important structural performance rather than optimizing a single aggregate score.

## Release discipline

Every candidate must retain source attribution, document its exact training provenance, keep evaluation data separate from training, and pass both quantitative gates and interactive handwriting checks. v1 will remain the public baseline until a successor improves broad validation without unacceptable regressions in structural cases or the fixed regression suite.
