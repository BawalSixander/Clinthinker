import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import database, main


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        storage = Path(self.temporary.name)
        patches = (
            patch.object(database, "STORAGE", storage),
            patch.object(database, "UPLOADS", storage / "uploads"),
            patch.object(database, "DB_PATH", storage / "assistant.sqlite3"),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        database.initialize()
        self.client_context = TestClient(main.app)
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)

    def test_check_reports_only_newer_numeric_versions(self):
        manifest = {
            "version": "2.10.0",
            "installer_url": "https://example.test/Clinthinker-Setup.exe",
            "changelog": "What's new",
        }
        with patch(
            "app.main.urllib.request.urlopen",
            return_value=io.BytesIO(json.dumps(manifest).encode("utf-8")),
        ) as open_url:
            response = self.client.get("/api/updates/check")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"available": True, **manifest})
        open_url.assert_called_once_with(main.UPDATE_MANIFEST_URL, timeout=5)

    def test_check_silently_ignores_equal_invalid_or_unreachable_manifests(self):
        manifests = (
            {"version": main.APP_VERSION, "installer_url": "https://example.test/setup.exe"},
            {"version": "2.0", "installer_url": "https://example.test/setup.exe"},
        )
        for manifest in manifests:
            with self.subTest(manifest=manifest), patch(
                "app.main.urllib.request.urlopen",
                return_value=io.BytesIO(json.dumps(manifest).encode("utf-8")),
            ):
                response = self.client.get("/api/updates/check")
                self.assertEqual(response.json(), {"available": False})

        with patch(
            "app.main.urllib.request.urlopen",
            side_effect=TimeoutError("offline"),
        ):
            response = self.client.get("/api/updates/check")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"available": False})

    def test_install_downloads_installer_backs_up_database_and_launches_it(self):
        database.DB_PATH.write_bytes(b"local database")
        with (
            patch("app.main.urllib.request.urlopen", return_value=io.BytesIO(b"installer")) as open_url,
            patch("app.main.subprocess.Popen") as launch,
        ):
            response = self.client.post(
                "/api/updates/install",
                json={"installer_url": "https://example.test/Clinthinker-Setup.exe"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"started": True})
        open_url.assert_called_once_with(
            "https://example.test/Clinthinker-Setup.exe",
            timeout=30,
        )
        backups = list(database.STORAGE.glob("assistant-*.sqlite3"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"local database")
        launch.assert_called_once()
        self.assertTrue(Path(launch.call_args.args[0][0]).is_file())


if __name__ == "__main__":
    unittest.main()
