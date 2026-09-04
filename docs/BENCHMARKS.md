# Benchmarks and interpretation

v1 was selected because its final-stage checkpoint recorded the strongest broad MathWriting validation score among the retained candidate set. The final report recorded 46.3634% exact match and 0.14817 mean token error across all 15,674 MathWriting validation expressions.

The same report recorded 21.0366% exact match and 0.53749 mean token error on 984 CROHME 2014 validation expressions, plus 94.54% exact match and 0.00245 mean token error on 10,000 synthetic piecewise validations.

The model also passed a fixed 12-case NewNotes regression suite spanning operators, integrals, sine, sequences, fractions, and piecewise expressions. A separate 260-case structural challenge was used diagnostically; it showed that superscript/subscript errors were still a major weakness. Treat these internal regression fixtures as regression checks, not as a general benchmark.

When comparing future releases, report at least:

- exact match and mean token error on the untouched MathWriting validation split;
- exact match and mean token error on an independently sourced writer/split;
- per-category scores for scripts, fractions, roots, matrices, integrals, long expressions, and cases;
- inference latency and memory on the target device; and
- raw failure examples only when their handwriting may be shared ethically and lawfully.
