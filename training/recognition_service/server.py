"""Local HTTP inference service for NewNotes recognition development."""

from __future__ import annotations

import argparse
import json
import os
from threading import Lock
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .recognizer import MODEL_VERSION, recognize_payload
from .symbol_model import load_model
from .image_latex import (
    DEFAULT_IMAGE_MODEL,
    ImageLatexModel,
    ImageModelUnavailable,
    PageImageLatexModel,
    PageImageModelUnavailable,
)


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


class RecognitionHandler(BaseHTTPRequestHandler):
    server_version = "NewNotesRecognition/0.1"
    model: Any = None
    image_model: Any = None
    # Image recognition is optional.  Keep it disabled unless the launcher
    # explicitly supplies a local checkpoint or an opt-in model identifier.
    image_model_path: str | None = None
    image_model_device = "auto"
    image_model_error: str | None = None
    image_model_lock = Lock()
    page_image_model: Any = None
    page_image_model_path: str | None = None
    page_image_model_device = "auto"
    page_image_model_error: str | None = None
    page_image_model_lock = Lock()
    feedback_path: Path | None = None
    feedback_lock = Lock()

    def _send_json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
        body = _json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_json(HTTPStatus.NO_CONTENT, {})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            handler_class = type(self)
            if handler_class.model is not None and not isinstance(handler_class.model, dict):
                model_version = handler_class.model.model_version
                device = str(handler_class.model.device)
            else:
                model_version = handler_class.model.get("modelVersion", "unknown") if handler_class.model else MODEL_VERSION
                device = "cpu"
            self._send_json(HTTPStatus.OK, {
                "status": "ok",
                "schemaVersion": 1,
                "modelVersion": model_version,
                "device": device,
                "imageModel": handler_class.image_model.model_version if handler_class.image_model is not None else handler_class.image_model_path or "disabled",
                "imageModelLoaded": handler_class.image_model is not None,
                "imageModelError": handler_class.image_model_error,
                "pageImageModel": handler_class.page_image_model.model_version if handler_class.page_image_model is not None else handler_class.page_image_model_path or "disabled",
                "pageImageModelLoaded": handler_class.page_image_model is not None,
                "pageImageModelError": handler_class.page_image_model_error,
                "feedbackCollection": handler_class.feedback_path is not None,
            })
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/v1/feedback":
            self._receive_feedback()
            return
        if self.path == "/v1/recognize-image":
            self._recognize_image()
            return
        if self.path != "/v1/recognize":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or payload.get("schemaVersion") != 1:
                raise ValueError("schemaVersion must be 1")
            response = recognize_payload(payload, self.model)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request", "message": str(error)})
            return
        self._send_json(HTTPStatus.OK, response)

    def _receive_feedback(self) -> None:
        handler_class = type(self)
        if handler_class.feedback_path is None:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {
                "error": "feedback_collection_disabled",
                "message": "Recognition feedback collection is not configured.",
            })
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 5 * 1024 * 1024:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_feedback", "message": "Feedback payload is empty or too large."})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or payload.get("schemaVersion") != 1 or payload.get("source") != "newnotes":
                raise ValueError("schemaVersion and source are required")
            if payload.get("eventType") not in {"accepted", "corrected"}:
                raise ValueError("eventType must be accepted or corrected")
            if not isinstance(payload.get("strokes"), list) or not isinstance(payload.get("modelSuggestion"), dict) or not isinstance(payload.get("finalLatex"), str):
                raise ValueError("strokes, modelSuggestion, and finalLatex are required")
            path = handler_class.feedback_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with handler_class.feedback_lock:
                with path.open("a", encoding="utf-8") as stream:
                    json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
                    stream.write("\n")
        except (ValueError, TypeError, json.JSONDecodeError, OSError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_feedback", "message": str(error)})
            return
        self._send_json(HTTPStatus.ACCEPTED, {"schemaVersion": 1, "accepted": True})

    def _recognize_image(self) -> None:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if not content_type.startswith("image/"):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_image", "message": "Send the image bytes with an image/* Content-Type."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 20 * 1024 * 1024:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_image", "message": "Image must be between 1 byte and 20 MB."})
            return
        try:
            handler_class = type(self)
            if not handler_class.image_model_path and handler_class.image_model is None and not handler_class.page_image_model_path and handler_class.page_image_model is None:
                raise ImageModelUnavailable(
                    "Image to LaTeX is optional and is not configured for this service. "
                    "Start it with --image-model pointing to a local model directory."
                )
            if handler_class.page_image_model_path and handler_class.page_image_model is None:
                with handler_class.page_image_model_lock:
                    if handler_class.page_image_model is None:
                        handler_class.page_image_model = PageImageLatexModel(
                            handler_class.page_image_model_path,
                            handler_class.page_image_model_device,
                        )
                        handler_class.page_image_model_error = None
            if handler_class.page_image_model is not None:
                response = handler_class.page_image_model.recognize(self.rfile.read(length))
            else:
                if handler_class.image_model is None:
                    with handler_class.image_model_lock:
                        if handler_class.image_model is None:
                            handler_class.image_model = ImageLatexModel(
                                handler_class.image_model_path,
                                handler_class.image_model_device,
                            )
                            handler_class.image_model_error = None
                response = handler_class.image_model.recognize(self.rfile.read(length))
        except (ImageModelUnavailable, PageImageModelUnavailable) as error:
            type(self).image_model_error = str(error)
            type(self).page_image_model_error = str(error)
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "image_model_unavailable", "message": str(error)})
            return
        except (ValueError, OSError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_image", "message": str(error)})
            return
        except Exception as error:  # keep model failures from killing the local service
            type(self).image_model_error = str(error)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "image_recognition_failed", "message": str(error)})
            return
        self._send_json(HTTPStatus.OK, {"schemaVersion": 1, "source": "image", **response})

    def log_message(self, format: str, *args: Any) -> None:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NewNotes local recognition service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--expression-model", type=Path, help="Optional PyTorch full-expression checkpoint.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--image-model",
        default=os.environ.get("NEWNOTES_IMAGE_MODEL_PATH") or None,
        help="Optional local image-to-LaTeX model directory; omitted means the feature is disabled.",
    )
    parser.add_argument(
        "--page-image-model",
        default=os.environ.get("NEWNOTES_PAGE_IMAGE_MODEL_PATH") or None,
        help="Optional page-level image model identifier or local path; this detects multiple formulas per image.",
    )
    parser.add_argument(
        "--feedback-path",
        default=os.environ.get("NEWNOTES_FEEDBACK_PATH") or None,
        help="Optional JSONL path for opt-in accepted or corrected recognition results.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    RecognitionHandler.image_model_path = args.image_model
    RecognitionHandler.image_model_device = args.device
    RecognitionHandler.page_image_model_path = args.page_image_model
    RecognitionHandler.page_image_model_device = args.device
    RecognitionHandler.feedback_path = Path(args.feedback_path) if args.feedback_path else None
    expression_path = args.expression_model or (Path(os.environ["NEWNOTES_EXPRESSION_MODEL_PATH"]) if os.environ.get("NEWNOTES_EXPRESSION_MODEL_PATH") else None)
    if expression_path:
        from .expression_inference import load_checkpoint

        RecognitionHandler.model = load_checkpoint(expression_path, args.device)
        print(f"Loaded expression checkpoint: {expression_path} on {RecognitionHandler.model.device}")
    default_model_path = Path(__file__).resolve().parents[1] / "artifacts" / "mathwriting-symbol-model.json"
    model_path = os.environ.get("NEWNOTES_SYMBOL_MODEL_PATH") or (str(default_model_path) if default_model_path.exists() else None)
    if model_path and RecognitionHandler.model is None:
        RecognitionHandler.model = load_model(model_path)
        print(f"Loaded recognition model: {model_path}")
    server = ThreadingHTTPServer((args.host, args.port), RecognitionHandler)
    print(f"NewNotes recognition service listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
