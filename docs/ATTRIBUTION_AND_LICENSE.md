# Attribution and release basis

## Release license

The model artifacts, reference implementation, and documentation in this repository are released under [Apache-2.0](../LICENSE).

This model is a transformative online-ink recognition system: it converts live pen-stroke sequences into LaTeX, does not redistribute raw source data, and is not a substitute for the source datasets. This release preserves acknowledgement of the source datasets while distributing the independently trained model and its runtime materials.

## Training-source acknowledgements

No raw MathWriting or CROHME files are included in this repository.

- **MathWriting Dataset** — Gervais, Fadeeva, and Maksai, *MathWriting: A Dataset For Handwritten Mathematical Expression Recognition*. The project used MathWriting online-ink data and its official train/validation boundaries. Source: <https://github.com/google-research/google-research/tree/master/mathwriting>.
- **CROHME** — the CROHME 2011–2014 online handwritten mathematical expression collection, using 2011–2013 expression-training data and 2014 expression validation. Source: <https://tc11.cvc.uab.es/datasets/CROHME-2014_2>.
- **Synthetic piecewise corpus** — generated locally from MathWriting glyph strokes. It contains controlled recombinations and geometry variation, not raw records from a separate corpus and not a claim of new-writer diversity.

Preserve these acknowledgements when redistributing v1 or publishing a derivative. Cite the original dataset authors in research outputs.
