from __future__ import annotations

import re
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import shutil

from fastapi import HTTPException, UploadFile

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_MATERIAL_TYPES = {".pdf", ".docx", ".txt"}
OCR_MIN_PAGE_TEXT = 40
MAX_OCR_PAGES = 100


async def read_limited(upload: UploadFile, maximum: int = MAX_UPLOAD_BYTES) -> bytes:
    content = await upload.read(maximum + 1)
    if len(content) > maximum:
        raise HTTPException(status_code=413, detail="File exceeds the 20 MB limit.")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return content


def extract_text(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_MATERIAL_TYPES:
        raise HTTPException(
            status_code=415, detail="Only PDF, DOCX, and TXT files are supported."
        )
    try:
        if extension == ".txt":
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = content.decode("cp1252")
        elif extension == ".docx":
            from docx import Document

            document = Document(BytesIO(content))
            paragraphs = [paragraph.text for paragraph in document.paragraphs]
            paragraphs.extend(
                cell.text
                for table in document.tables
                for row in table.rows
                for cell in row.cells
            )
            text = "\n".join(paragraphs)
        else:
            import pdfplumber

            with pdfplumber.open(BytesIO(content)) as pdf:
                page_texts = [page.extract_text() or "" for page in pdf.pages]
            sparse_pages = [
                index for index, page_text in enumerate(page_texts)
                if len(page_text.strip()) < OCR_MIN_PAGE_TEXT
            ]
            if len(sparse_pages) > MAX_OCR_PAGES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"{filename} has more than {MAX_OCR_PAGES} pages that need OCR. "
                        "Split the document into smaller files and try again."
                    ),
                )
            if sparse_pages:
                import pypdfium2
                import pytesseract

                document = pypdfium2.PdfDocument(content)
                try:
                    ocr = _get_ocr_engine(pytesseract)
                    for index in sparse_pages:
                        image = document[index].render(scale=2).to_pil()
                        page_texts[index] = ocr.image_to_string(image, lang="eng")
                finally:
                    document.close()
            text = "\n".join(page_texts)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=422, detail=f"Could not extract text from {filename}: {error}"
        ) from error
    if not text.strip():
        raise HTTPException(
            status_code=422, detail=f"No readable text was found in {filename}."
        )
    return text


def extract_lesson_features(filename: str, content: bytes) -> dict[str, list[str]]:
    extension = Path(filename).suffix.lower()
    features = {"emphasized_terms": [], "heading_terms": [], "list_terms": []}
    if extension == ".docx":
        from docx import Document

        document = Document(BytesIO(content))
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            style_name = paragraph.style.name.casefold() if paragraph.style else ""
            if style_name.startswith("heading") or "title" in style_name:
                features["heading_terms"].append(text)
            if "list" in style_name:
                features["list_terms"].append(text)
            for run in paragraph.runs:
                if run.bold or run.italic:
                    features["emphasized_terms"].append(run.text)
    elif extension == ".pdf":
        import pdfplumber

        with pdfplumber.open(BytesIO(content)) as pdf:
            for page in pdf.pages:
                words = page.extract_words(extra_attrs=["fontname", "size"])
                sizes = sorted(word["size"] for word in words if word.get("text", "").strip())
                normal_size = sizes[len(sizes) // 2] if sizes else 0
                lines: dict[float, list[dict]] = {}
                for word in words:
                    lines.setdefault(round(word["top"]), []).append(word)
                    if re.search(r"bold|black|heavy|demi", word.get("fontname", ""), re.I):
                        features["emphasized_terms"].append(word["text"])
                for line_words in lines.values():
                    line = " ".join(word["text"] for word in line_words).strip()
                    line_size = max((word["size"] for word in line_words), default=0)
                    if line and normal_size and line_size >= normal_size * 1.15:
                        features["heading_terms"].append(line)
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("cp1252")
        for line in text.splitlines():
            stripped = line.strip()
            if re.match(r"^#{1,6}\s+", stripped) or (
                2 <= len(stripped.split()) <= 12
                and stripped.isupper()
                and len(stripped) <= 100
            ):
                features["heading_terms"].append(re.sub(r"^#{1,6}\s+", "", stripped))
            if re.match(r"^(?:[-*•]|\d+[.)])\s+", stripped):
                features["list_terms"].append(
                    re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", stripped)
                )
            features["emphasized_terms"].extend(re.findall(r"\*\*([^*]+)\*\*", stripped))
    return features


@lru_cache(maxsize=1)
def _get_ocr_engine(pytesseract):
    candidates = [
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    executable = next((candidate for candidate in candidates if candidate and Path(candidate).is_file()), None)
    if not executable:
        raise HTTPException(
            status_code=422,
            detail=(
                "This PDF appears to contain scanned pages. Install Tesseract OCR "
                "with English language data, add tesseract.exe to PATH, and restart the app."
            ),
        )
    pytesseract.pytesseract.tesseract_cmd = executable

    return pytesseract
