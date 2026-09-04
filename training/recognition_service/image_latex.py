"""Optional image-to-LaTeX inference using a Hugging Face vision model.

The image path is intentionally lazy and isolated from handwriting inference.
The normal NewNotes service can still start when the optional image packages
are not installed; image requests then return an actionable 503 response.
"""

from __future__ import annotations

import io
import re
import argparse
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_IMAGE_MODEL = "wanderkid/unimernet_small"
DEFAULT_PAGE_IMAGE_MODEL = "OleehyO/TexTeller"


class ImageModelUnavailable(RuntimeError):
    """Raised when optional image-recognition dependencies or weights are absent."""


class PageImageModelUnavailable(RuntimeError):
    """Raised when the page-level formula detector is unavailable."""


def normalize_latex(value: str) -> str:
    """Remove common presentation wrappers without changing math structure."""

    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:latex|tex)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    if text.startswith("\\[") and text.endswith("\\]"):
        text = text[2:-2].strip()
    if len(text) >= 2 and text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()
    # TexTeller can emit an empty delimiter pair after a tightly cropped
    # display equation. It has no visual meaning and only creates a stray
    # delimiter in the MathLive box.
    text = re.sub(r"\s+\\left\(\s*\\right\.\s*$", "", text)
    return text


class ImageLatexModel:
    """Lazy UniMERNet image-to-LaTeX model wrapper.

    The small/base/tiny UniMERNet repositories publish custom ``*.pth``
    checkpoints, not Transformers ``pytorch_model.bin`` files. Those models
    must use UniMERNet's official config/task loader; newer HF-converted
    checkpoints can still use the generic Transformers path below.
    """

    def __init__(self, model_path: str = DEFAULT_IMAGE_MODEL, device: str = "auto") -> None:
        try:
            import torch
            from PIL import Image, ImageOps
            from transformers import AutoProcessor
        except ImportError as error:
            raise ImageModelUnavailable(
                "Image recognition requires Pillow and transformers. "
                "Install recognition-service/requirements-image-ml.txt."
            ) from error

        self._torch = torch
        self._image_class = Image
        self._official_unimernet = False
        self.device = self._choose_device(torch, device)
        try:
            if self._is_custom_unimernet_checkpoint(model_path):
                self._load_official_unimernet(model_path)
            else:
                self._load_transformers_model(model_path, AutoProcessor)
        except Exception as error:  # model downloads and remote configs vary by release
            raise ImageModelUnavailable(f"Unable to load image model '{model_path}': {error}") from error
        self.model_version = model_path

    @staticmethod
    def _is_custom_unimernet_checkpoint(model_path: str) -> bool:
        local_path = Path(model_path).expanduser()
        if local_path.is_dir():
            return any(local_path.glob("*.pth"))
        return model_path.startswith("wanderkid/unimernet_")

    def _load_transformers_model(self, model_path: str, auto_processor: Any) -> None:
        try:
            from transformers import AutoModelForImageTextToText as model_class
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as model_class
            except ImportError as error:
                raise ImageModelUnavailable(
                    "This transformers version does not provide an image-to-text model loader."
                ) from error
        self.processor = auto_processor.from_pretrained(model_path, trust_remote_code=True)
        self.model = model_class.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    def _load_official_unimernet(self, model_path: str) -> None:
        self._register_torchvision_compat()
        self._register_transformers_compat()
        try:
            from huggingface_hub import snapshot_download
            from unimernet.common.config import Config
            import unimernet.tasks as tasks
            from unimernet.processors import load_processor
        except ImportError as error:
            raise ImageModelUnavailable(
                "The selected UniMERNet checkpoint uses the official custom loader. "
                "Install recognition-service/requirements-image-ml.txt first."
            ) from error

        local_path = Path(model_path).expanduser()
        if not local_path.is_dir():
            local_path = Path(snapshot_download(model_path))
        checkpoint = next(local_path.glob("*.pth"), None)
        if checkpoint is None:
            raise ImageModelUnavailable(f"No UniMERNet .pth checkpoint found in {local_path}.")
        model_name = local_path.as_posix()
        checkpoint_path = checkpoint.as_posix()
        config_text = f"""
model:
  arch: unimernet
  model_type: unimernet
  model_config:
    model_name: {model_name!r}
    max_seq_len: 1536
  load_pretrained: True
  pretrained: {checkpoint_path!r}
  tokenizer_config:
    path: {model_name!r}
datasets:
  formula_rec_eval:
    vis_processor:
      eval:
        name: formula_image_eval
        image_size: [192, 672]
run:
  runner: runner_iter
  task: unimernet_train
  batch_size_train: 1
  batch_size_eval: 1
  num_workers: 1
  iters_per_inner_epoch: 1
  max_iters: 1
  seed: 42
  output_dir: {model_name!r}
  evaluate: False
  device: {str(self.device)!r}
  world_size: 1
  distributed: False
  generate_cfg:
    temperature: 0.0
    do_sample: False
"""
        config_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", encoding="utf-8", delete=False,
        )
        try:
            config_file.write(config_text)
            config_file.close()
            args = argparse.Namespace(cfg_path=config_file.name, options=None)
            config = Config(args)
            task = tasks.setup_task(config)
            self.model = task.build_model(config).to(self.device)
            self.processor = load_processor(
                "formula_image_eval",
                config.config.datasets.formula_rec_eval.vis_processor.eval,
            )
            self.model.eval()
            self._official_unimernet = True
        finally:
            Path(config_file.name).unlink(missing_ok=True)

    def _register_torchvision_compat(self) -> None:
        """Allow torchvision's metadata decorators with newer Torch builds.

        Some CUDA wheels ship the torchvision Python metadata before the
        optional C++ NMS operators are registered. UniMERNet only needs the
        dataset/transforms portions of torchvision, so defining the metadata
        operator names is a safe compatibility bridge for that combination.
        """

        try:
            library = self._torch.library.Library("torchvision", "DEF")
            library.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
            library.define("qnms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
            self._torchvision_compat_library = library
        except Exception:
            # Older/matching torchvision builds already provide these ops.
            self._torchvision_compat_library = None

    def _register_transformers_compat(self) -> None:
        """Bridge utility imports removed from newer Transformers releases."""

        try:
            import transformers.modeling_utils as modeling_utils
            import transformers.pytorch_utils as pytorch_utils
        except ImportError:
            return

        for name in ("apply_chunking_to_forward", "prune_linear_layer"):
            if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
                setattr(modeling_utils, name, getattr(pytorch_utils, name))
            if not hasattr(pytorch_utils, name) and hasattr(modeling_utils, name):
                setattr(pytorch_utils, name, getattr(modeling_utils, name))

        if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
            setattr(
                modeling_utils,
                "find_pruneable_heads_and_indices",
                self._find_pruneable_heads_and_indices,
            )
        if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
            setattr(
                pytorch_utils,
                "find_pruneable_heads_and_indices",
                self._find_pruneable_heads_and_indices,
            )

    def _find_pruneable_heads_and_indices(
        self,
        heads: Any,
        n_heads: int,
        head_size: int,
        already_pruned_heads: Any,
    ) -> tuple[set[int], Any]:
        """Compatibility copy of Transformers' legacy attention helper."""

        mask = self._torch.ones(n_heads, head_size)
        heads = set(heads) - set(already_pruned_heads)
        for head in heads:
            head -= sum(1 for pruned_head in already_pruned_heads if pruned_head < head)
            mask[head] = 0
        index = self._torch.arange(len(mask.view(-1)))[mask.view(-1).bool()]
        return heads, index

    @staticmethod
    def _choose_device(torch: Any, requested: str) -> str:
        if requested == "cuda" and not torch.cuda.is_available():
            raise ImageModelUnavailable("CUDA was requested for image recognition but is unavailable.")
        if requested in {"cuda", "cpu"}:
            return requested
        return "cuda" if torch.cuda.is_available() else "cpu"

    def recognize(self, image_bytes: bytes) -> dict[str, Any]:
        try:
            image = self._image_class.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as error:
            raise ValueError(f"Unable to decode the image: {error}") from error

        if not self._official_unimernet:
            inputs = self.processor(images=image, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            with self._torch.inference_mode():
                output_ids = self.model.generate(**inputs, max_new_tokens=512)
            decoded = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        else:
            inputs = self.processor(image).unsqueeze(0).to(self.device)
            with self._torch.inference_mode():
                output = self.model.generate({"image": inputs})
            decoded = output["pred_str"][0]
        latex = normalize_latex(decoded)
        if not latex:
            raise ValueError("The image model returned an empty LaTeX expression.")
        return {
            "latex": latex,
            "display": latex,
            # UniMERNet does not expose a calibrated confidence score through
            # its standard generation API. Keep this visibly conservative.
            "confidence": 0.5,
            "modelVersion": self.model_version,
        }


class PageImageLatexModel:
    """Detect and recognize multiple formulas in a photographed page.

    TexTeller's page detector is used to find formula regions first. The
    recognizer then receives each cropped region independently, which avoids
    asking a single-expression model to interpret an entire notebook page,
    whiteboard photo, or application screenshot.

    The detector intentionally runs through ONNX Runtime's CPU provider. The
    formula recognizer still uses the requested Torch device; this keeps the
    feature usable with the CPU-only ONNX Runtime package while preserving CUDA
    for the expensive sequence model.
    """

    def __init__(self, model_path: str = DEFAULT_PAGE_IMAGE_MODEL, device: str = "auto") -> None:
        try:
            import numpy as np
            import torch
            from PIL import Image, ImageOps
            from onnxruntime import InferenceSession
            from texteller.api.detection import latex_detect
            from texteller.api.inference import img2latex
            from texteller.api.load import _maybe_download
            from texteller.constants import LATEX_DET_MODEL_URL
            from texteller import load_model, load_tokenizer
        except ImportError as error:
            raise PageImageModelUnavailable(
                "Page image recognition requires TexTeller, ONNX Runtime, and NumPy. "
                "Install recognition-service/requirements-image-ml.txt."
            ) from error

        self._np = np
        self._image_class = Image
        self._image_ops = ImageOps
        self._torch = torch
        self._latex_detect = latex_detect
        self._img2latex = img2latex
        self.device = self._choose_device(torch, device)
        try:
            detector_path = _maybe_download(LATEX_DET_MODEL_URL)
            self.detector = InferenceSession(
                str(detector_path),
                providers=["CPUExecutionProvider"],
            )
            is_default_model = model_path in {
                DEFAULT_PAGE_IMAGE_MODEL,
                "OleehyO/TexTeller_en",
            }
            model_source = None if is_default_model else model_path
            self.model = load_model(model_source, use_onnx=False)
            self.tokenizer = load_tokenizer(model_source)
            self.model.to(self.device)
            self.model.eval()
        except Exception as error:
            raise PageImageModelUnavailable(
                f"Unable to load page image model '{model_path}': {error}"
            ) from error
        self.model_version = model_path

    @staticmethod
    def _choose_device(torch: Any, requested: str) -> Any:
        if requested == "cuda" and not torch.cuda.is_available():
            raise PageImageModelUnavailable(
                "CUDA was requested for page image recognition but is unavailable."
            )
        if requested == "cuda":
            return torch.device("cuda")
        if requested == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def recognize(self, image_bytes: bytes) -> dict[str, Any]:
        try:
            image = self._image_class.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as error:
            raise ValueError(f"Unable to decode the image: {error}") from error

        width, height = image.size
        from tempfile import NamedTemporaryFile

        # TexTeller's detector is trained primarily on dark ink over a light
        # page. NewNotes also accepts screenshots/chalkboard captures, where
        # the same formula is light ink over a dark background. Run the
        # detector on an inverted variant for those inputs and keep whichever
        # pass produces the stronger spatial detections.
        image_array = self._np.asarray(image)
        border_size = max(1, min(width, height) // 20)
        border_pixels = self._np.concatenate(
            (
                image_array[:border_size].reshape(-1, 3),
                image_array[-border_size:].reshape(-1, 3),
                image_array[:, :border_size].reshape(-1, 3),
                image_array[:, -border_size:].reshape(-1, 3),
            ),
            axis=0,
        )
        background_luminance = float(border_pixels.mean())
        inverted_image = self._image_ops.invert(image) if background_luminance < 140 else None

        def detection_score(items: list[Any]) -> float:
            return sum(
                max(0, int(item.w))
                * max(0, int(item.h))
                * max(0.0, float(item.confidence or 0.0))
                for item in items
            )

        temporary_paths: list[Path] = []
        try:
            with NamedTemporaryFile(suffix=".png", delete=False) as temporary:
                temporary.write(image_bytes)
                temporary_path = Path(temporary.name)
                temporary_paths.append(temporary_path)
            boxes = self._latex_detect(str(temporary_path), self.detector)
            recognition_image = image

            if inverted_image is not None:
                with NamedTemporaryFile(suffix=".png", delete=False) as inverted_file:
                    inverted_path = Path(inverted_file.name)
                    temporary_paths.append(inverted_path)
                inverted_image.save(inverted_path)
                inverted_boxes = self._latex_detect(str(inverted_path), self.detector)
                if detection_score(inverted_boxes) > detection_score(boxes) * 1.5:
                    boxes = inverted_boxes
                    recognition_image = inverted_image
        finally:
            for path in temporary_paths:
                path.unlink(missing_ok=True)

        valid_boxes = []
        for box in sorted(boxes):
            x = max(0, min(width - 1, int(box.p.x)))
            y = max(0, min(height - 1, int(box.p.y)))
            right = max(x + 1, min(width, x + int(box.w)))
            bottom = max(y + 1, min(height, y + int(box.h)))
            if right - x >= 12 and bottom - y >= 12:
                valid_boxes.append((x, y, right, bottom, float(box.confidence or 0.5)))

        if not valid_boxes:
            raise ValueError(
                "No formula regions were detected. Try a clearer photo with the math in focus."
            )

        image_array = self._np.asarray(recognition_image).copy()
        crops = [image_array[y:bottom, x:right] for x, y, right, bottom, _ in valid_boxes]
        with self._torch.inference_mode():
            decoded = self._img2latex(
                model=self.model,
                tokenizer=self.tokenizer,
                images=crops,
                device=self.device,
                out_format="latex",
                keep_style=False,
                num_beams=1,
            )

        regions = []
        for (x, y, right, bottom, detector_confidence), latex in zip(valid_boxes, decoded):
            normalized = normalize_latex(latex)
            if not normalized:
                continue
            regions.append({
                "x": x,
                "y": y,
                "width": right - x,
                "height": bottom - y,
                "latex": normalized,
                "display": normalized,
                "confidence": max(0.0, min(1.0, detector_confidence)),
            })

        if not regions:
            raise ValueError("The page model detected formulas but returned no LaTeX.")
        return {
            "mode": "page",
            "imageWidth": width,
            "imageHeight": height,
            "regions": regions,
            "latex": " ".join(region["latex"] for region in regions),
            "display": " ".join(region["display"] for region in regions),
            "confidence": min(region["confidence"] for region in regions),
            "modelVersion": self.model_version,
        }
