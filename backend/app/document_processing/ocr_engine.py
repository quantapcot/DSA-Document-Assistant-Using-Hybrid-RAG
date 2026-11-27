"""B1: OCR engine — trích xuất text từ ảnh và PDF scan (EasyOCR primary, Tesseract fallback)."""

from __future__ import annotations

import io
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import settings

logger = logging.getLogger("app.document_processing.ocr_engine")

DEFAULT_LANGUAGES: tuple[str, ...] = ("vi", "en")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL_DIR = _REPO_ROOT / "data" / "runtime" / "ocr_models"
_REQUIRED_EASYOCR_MODELS = ("craft_mlt_25k.pth", "latin_g2.pth")
_OCR_UNAVAILABLE_WARNED = False


def _parse_languages(languages: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if not languages:
        raw = settings.OCR_LANGUAGES
        items = [part.strip() for part in raw.split(",") if part.strip()]
        return tuple(items) if items else DEFAULT_LANGUAGES
    if isinstance(languages, str):
        items = [part.strip() for part in languages.split(",") if part.strip()]
        return tuple(items) if items else DEFAULT_LANGUAGES
    items = [str(part).strip() for part in languages if str(part).strip()]
    return tuple(items) if items else DEFAULT_LANGUAGES


def _has_cached_easyocr_models(directory: Path) -> bool:
    return directory.is_dir() and all((directory / name).is_file() for name in _REQUIRED_EASYOCR_MODELS)


def _is_dir_writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _resolve_model_storage() -> tuple[str | None, bool]:
    """Chọn thư mục lưu weights EasyOCR an toàn cho cả sandbox, local và Docker."""
    configured = settings.OCR_MODEL_DIR or os.environ.get("OCR_MODEL_DIR")
    if configured:
        model_dir = Path(configured)
        if _has_cached_easyocr_models(model_dir):
            return str(model_dir), False
        model_dir.mkdir(parents=True, exist_ok=True)
        return str(model_dir), True

    if _has_cached_easyocr_models(_DEFAULT_MODEL_DIR):
        return str(_DEFAULT_MODEL_DIR), False

    home_model_dir = Path.home() / ".EasyOCR" / "model"
    if _has_cached_easyocr_models(home_model_dir) and os.access(home_model_dir, os.R_OK):
        return str(home_model_dir), False

    if _is_dir_writable(home_model_dir):
        return None, True

    _DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return str(_DEFAULT_MODEL_DIR), True


@lru_cache(maxsize=4)
def _get_easyocr_reader(
    languages: tuple[str, ...],
    gpu: bool,
    model_dir: str | None,
    download_enabled: bool,
) -> Any:
    import easyocr

    logger.info(
        "Initializing EasyOCR reader (langs=%s, gpu=%s, model_dir=%s)",
        list(languages),
        gpu,
        model_dir or "default",
    )
    return easyocr.Reader(
        list(languages),
        gpu=gpu,
        model_storage_directory=model_dir,
        download_enabled=download_enabled,
        verbose=False,
    )


def _detect_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _to_rgb_pil(image_input: bytes | bytearray | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    elif isinstance(image_input, Path):
        img = Image.open(image_input)
    elif isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input))
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input).__name__}")

    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _prepare_image_array(
    image_input: bytes | bytearray | Path | Image.Image | np.ndarray,
    *,
    max_side: int | None = None,
) -> tuple[Image.Image, np.ndarray]:
    img = _to_rgb_pil(image_input)
    limit = max_side or settings.OCR_MAX_IMAGE_SIDE
    w, h = img.size

    if max(w, h) > limit:
        scale = limit / float(max(w, h))
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    elif h < 80 and w < 600:
        # Upscale nhỏ giúp OCR chữ cỡ bé trên screenshot sắc nét hơn
        img = img.resize((w * 2, h * 2), Image.Resampling.BICUBIC)

    return img, np.asarray(img)


def _sort_and_join_boxes(detections: list[Any], *, min_confidence: float) -> str:
    """Sắp xếp các bounding box theo thứ tự đọc tự nhiên (trên→dưới, trái→phải)."""
    items: list[tuple[float, float, float, str]] = []
    for det in detections:
        if not isinstance(det, (list, tuple)) or len(det) < 3:
            continue
        bbox, text, conf = det[0], str(det[1]).strip(), float(det[2])
        if not text or conf < min_confidence:
            continue
        xs = [float(pt[0]) for pt in bbox]
        ys = [float(pt[1]) for pt in bbox]
        y_top = min(ys)
        y_bottom = max(ys)
        y_center = (y_top + y_bottom) / 2.0
        line_h = max(8.0, y_bottom - y_top)
        x_left = min(xs)
        items.append((y_center, line_h, x_left, text))

    if not items:
        return ""

    # Gom các box cùng một dòng (chênh lệch y_center nhỏ hơn ~0.6 chiều cao dòng)
    items.sort(key=lambda item: (item[0], item[2]))
    lines: list[list[tuple[float, float, float, str]]] = []
    for item in items:
        if not lines:
            lines.append([item])
            continue
        last_line = lines[-1]
        avg_y = sum(x[0] for x in last_line) / len(last_line)
        avg_h = sum(x[1] for x in last_line) / len(last_line)
        if abs(item[0] - avg_y) <= max(10.0, avg_h * 0.6):
            last_line.append(item)
        else:
            lines.append([item])

    rendered_lines: list[str] = []
    for line in lines:
        line.sort(key=lambda item: item[2])
        rendered_lines.append(" ".join(item[3] for item in line))

    return "\n".join(rendered_lines).strip()


def _ocr_with_easyocr(arr: np.ndarray, languages: tuple[str, ...], min_confidence: float) -> str:
    model_dir, download_enabled = _resolve_model_storage()
    gpu = _detect_gpu()
    try:
        reader = _get_easyocr_reader(languages, gpu, model_dir, download_enabled)
        detections = reader.readtext(arr, detail=1, paragraph=False)
    except RuntimeError as exc:
        if gpu:
            logger.warning("EasyOCR GPU failed (%s); falling back to CPU.", exc)
            reader = _get_easyocr_reader(languages, False, model_dir, download_enabled)
            detections = reader.readtext(arr, detail=1, paragraph=False)
        else:
            raise
    return _sort_and_join_boxes(detections, min_confidence=min_confidence)


def _ocr_with_tesseract(pil_img: Image.Image, languages: tuple[str, ...]) -> str:
    import pytesseract

    lang_map = {"vi": "vie", "en": "eng"}
    tess_langs = "+".join(lang_map.get(lang, lang) for lang in languages)
    return pytesseract.image_to_string(pil_img, lang=tess_langs).strip()


def is_ocr_available() -> bool:
    """Kiểm tra có ít nhất một OCR backend khả dụng hay không."""
    try:
        import easyocr  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def extract_text_from_image(
    image_input: bytes | bytearray | Path | Image.Image | np.ndarray,
    *,
    languages: str | tuple[str, ...] | list[str] | None = None,
    min_confidence: float = 0.25,
    max_side: int | None = None,
) -> str:
    """
    Trích xuất văn bản từ ảnh bằng OCR (hỗ trợ tiếng Việt + tiếng Anh).

    Trả về chuỗi rỗng nếu `OCR_ENABLED=False`, ảnh hỏng, hoặc không có OCR backend.
    """
    global _OCR_UNAVAILABLE_WARNED

    if not settings.OCR_ENABLED:
        return ""

    langs = _parse_languages(languages)
    try:
        pil_img, arr = _prepare_image_array(image_input, max_side=max_side)
    except Exception as exc:
        logger.debug("Skipping unreadable image for OCR: %s", exc)
        return ""

    # 1. Primary: EasyOCR
    try:
        return _ocr_with_easyocr(arr, langs, min_confidence)
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("EasyOCR failed (%s); attempting fallback if available.", exc)

    # 2. Optional fallback: Tesseract
    try:
        return _ocr_with_tesseract(pil_img, langs)
    except Exception:
        if not _OCR_UNAVAILABLE_WARNED:
            logger.warning("No OCR backend available (install easyocr or pytesseract + tesseract-ocr).")
            _OCR_UNAVAILABLE_WARNED = True
        return ""

