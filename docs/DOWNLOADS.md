# Downloads and installation

## Runtime bundle: recommended for application developers

Open the [v1.0.0 release](https://github.com/Shrookin/real-time-handwriting-to-latex/releases/tag/v1.0.0) and download `handwriting-to-latex-v1.zip`. The ZIP contains everything needed for offline ONNX Runtime inference:

- `model/handwriting-to-latex-v1.onnx` — the runtime model;
- `model/handwriting-to-latex-v1-vocab.json` — required token mapping;
- `model/manifest.json` — input/output contract and SHA-256 hashes;
- `examples/run_onnx.py` and `examples/sample_request.json` — a working CPU reference;
- `requirements.txt`, `LICENSE`, `NOTICE`, and integration/model documentation.

From a terminal:

```bash
curl -L -o handwriting-to-latex-v1.zip \
  https://github.com/Shrookin/real-time-handwriting-to-latex/releases/download/v1.0.0/handwriting-to-latex-v1.zip
unzip handwriting-to-latex-v1.zip
cd handwriting-to-latex-v1
python -m venv .venv
python -m pip install -r requirements.txt
python examples/run_onnx.py --input examples/sample_request.json
```

On Windows PowerShell, use `Invoke-WebRequest` and `Expand-Archive` instead:

```powershell
Invoke-WebRequest 'https://github.com/Shrookin/real-time-handwriting-to-latex/releases/download/v1.0.0/handwriting-to-latex-v1.zip' -OutFile handwriting-to-latex-v1.zip
Expand-Archive .\\handwriting-to-latex-v1.zip -DestinationPath .
Set-Location .\\handwriting-to-latex-v1
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe examples\\run_onnx.py --input examples\\sample_request.json
```

## Full clone: recommended for research and retraining

```bash
git clone https://github.com/Shrookin/real-time-handwriting-to-latex.git
cd real-time-handwriting-to-latex
python -m pip install -r requirements.txt
python examples/run_onnx.py --input examples/sample_request.json
```

The full clone additionally includes the training, synthetic-corpus, evaluation, and ONNX-export source under `training/`. It intentionally excludes raw datasets, generated training caches, and collected handwriting.

## Add v1 to an application

For Python, use the included `examples/run_onnx.py` as the reference integration. For web, mobile, Rust, or an HTTP service, bundle or download the three runtime files in `model/`, implement the six-feature preprocessing exactly, execute the ONNX graph, and decode IDs through the vocabulary until `<eos>`. The details and input JSON contract are in [INTEGRATION.md](INTEGRATION.md).

Do not use the PyTorch `.pt` checkpoint for normal application inference; use the ONNX artifact and its matching vocabulary. Verify downloaded files against the hashes in `model/manifest.json` before loading them.
