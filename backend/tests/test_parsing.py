import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from app.parsing import extract_lesson_features, extract_text


class LessonFeatureTests(unittest.TestCase):
    def test_text_lesson_detects_headings_lists_and_emphasis(self):
        features = extract_lesson_features(
            "lesson.txt",
            (
                "# Network Protocols\n"
                "**Packet switching** is a method that sends data in small units.\n"
                "- A router forwards data packets between networks.\n"
            ).encode("utf-8"),
        )

        self.assertIn("Network Protocols", features["heading_terms"])
        self.assertIn("Packet switching", features["emphasized_terms"])
        self.assertIn(
            "A router forwards data packets between networks.",
            features["list_terms"],
        )


class PdfOcrTests(unittest.TestCase):
    def test_scanned_pdf_page_uses_local_ocr_fallback(self):
        page = MagicMock()
        page.extract_text.return_value = ""
        pdf = MagicMock()
        pdf.__enter__.return_value.pages = [page]
        image = object()
        rendered_page = MagicMock()
        rendered_page.render.return_value.to_pil.return_value = image
        pdfium_document = MagicMock()
        pdfium_document.__getitem__.return_value = rendered_page
        ocr = MagicMock()
        ocr.image_to_string.return_value = (
            "Photosynthesis is the process by which plants convert light energy."
        )

        with (
            patch("pdfplumber.open", return_value=pdf),
            patch("pypdfium2.PdfDocument", return_value=pdfium_document),
            patch("app.parsing._get_ocr_engine", return_value=ocr),
        ):
            text = extract_text("scanned.pdf", b"pdf bytes")

        self.assertIn("Photosynthesis is the process", text)
        ocr.image_to_string.assert_called_once_with(image, lang="eng")
        pdfium_document.close.assert_called_once()

    def test_scanned_pdf_without_tesseract_explains_required_setup(self):
        page = MagicMock()
        page.extract_text.return_value = ""
        pdf = MagicMock()
        pdf.__enter__.return_value.pages = [page]

        with (
            patch("pdfplumber.open", return_value=pdf),
            patch("pypdfium2.PdfDocument"),
            patch("shutil.which", return_value=None),
        ):
            with self.assertRaises(HTTPException) as error:
                extract_text("scanned.pdf", b"pdf bytes")

        self.assertEqual(error.exception.status_code, 422)
        self.assertIn("Install Tesseract OCR", error.exception.detail)


if __name__ == "__main__":
    unittest.main()
