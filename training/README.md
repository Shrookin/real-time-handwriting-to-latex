# Training source package

This directory contains the Python source used for the online-ink parser, fixed-shape cache builders, synthetic piecewise generation, mixed replay trainer, checkpoint evaluator, and ONNX export workflow. It contains no source datasets, caches, collected handwriting, or experiment artifacts.

Create an environment and install a PyTorch build appropriate for the target CPU/GPU before installing the remaining dependency:

```bash
python -m venv .venv
# Activate the virtual environment, then install a PyTorch wheel from https://pytorch.org/get-started/locally/
python -m pip install -r requirements-training.txt
```

Run modules from the repository root, for example:

```bash
python -m training.recognition_service.expression_mixed_train --help
python -m training.recognition_service.evaluate_expression_checkpoint --help
python -m training.recognition_service.generate_piecewise_corpus --help
```

Use the frozen model card and training documentation as the promotion contract. Preserve original source split boundaries, prevent training/evaluation leakage, evaluate every candidate on untouched data, and retain the generated run report with the output checkpoint.
