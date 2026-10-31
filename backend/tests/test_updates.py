import io
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from fastapi.testclient import TestClient

from app import database, main


class UpdateTests(unittest.TestCase):
    installer_url = (
        "https://github.com/seanlordtrillo-ops/Clinthinker/"
        "releases/download/v2.0.1/Clinthinker-Setup-2.0.1.exe"
    )

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
            "installer_url": self.installer_url,
            "changelog": "What's new",
        }
        with patch(
            "app.main.urllib.request.urlopen",
            return_value=io.BytesIO(json.dumps(manifest).encode("utf-8")),
        ) as open_url:
            response = self.client.get("/api/updates/check")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"available": True, "version": "2.10.0", "changelog": "What's new"},
        )
        open_url.assert_called_once_with(main.UPDATE_MANIFEST_URL, timeout=5)

    def test_check_shows_no_banner_when_manifest_version_matches(self):
        manifest = {"version": main.APP_VERSION}
        with patch(
            "app.main.urllib.request.urlopen",
            return_value=io.BytesIO(json.dumps(manifest).encode("utf-8")),
        ):
            response = self.client.get("/api/updates/check")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"available": False})

    def test_check_silently_ignores_equal_invalid_or_unreachable_manifests(self):
        manifests = (
            {"version": "2.0", "installer_url": "https://example.test/setup.exe"},
            {"version": "2.0.1", "installer_url": "https://github.com.attacker.test/setup.exe"},
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

    def test_install_uses_manifest_url_checks_sha_backs_up_and_launches(self):
        database.DB_PATH.write_bytes(b"local database")
        installer_bytes = b"installer package"
        manifest = {
            "version": "2.0.1",
            "installer_url": self.installer_url,
            "sha256": hashlib.sha256(installer_bytes).hexdigest(),
        }
        with (
            patch(
                "app.main.urllib.request.urlopen",
                side_effect=[
                    io.BytesIO(json.dumps(manifest).encode("utf-8")),
                    io.BytesIO(installer_bytes),
                ],
            ) as open_url,
            patch("app.main.subprocess.Popen") as launch,
        ):
            response = self.client.post("/api/updates/install")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"started": True})
        self.assertEqual(
            open_url.call_args_list,
            [
                call(main.UPDATE_MANIFEST_URL, timeout=5),
                call(self.installer_url, timeout=30),
            ],
        )
        backups = list(database.STORAGE.glob("assistant-*.sqlite3"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"local database")
        launch.assert_called_once()
        self.assertTrue(Path(launch.call_args.args[0][0]).is_file())

    def test_install_rejects_non_github_manifest_url(self):
        manifest = {
            "version": "2.0.1",
            "installer_url": "https://github.com.attacker.test/setup.exe",
        }
        with patch(
            "app.main.urllib.request.urlopen",
            return_value=io.BytesIO(json.dumps(manifest).encode("utf-8")),
        ) as open_url, patch("app.main.subprocess.Popen") as launch:
            response = self.client.post("/api/updates/install")

        self.assertEqual(response.status_code, 422)
        self.assertEqual(open_url.call_count, 1)
        launch.assert_not_called()

    def test_install_route_has_no_request_body(self):
        operation = self.client.get("/openapi.json").json()["paths"][
            "/api/updates/install"
        ]["post"]
        self.assertNotIn("requestBody", operation)

    def test_install_refuses_checksum_mismatch_before_backup_or_launch(self):
        database.DB_PATH.write_bytes(b"local database")
        manifest = {
            "version": "2.0.1",
            "installer_url": self.installer_url,
            "sha256": "0" * 64,
        }
        with patch(
            "app.main.urllib.request.urlopen",
            side_effect=[
                io.BytesIO(json.dumps(manifest).encode("utf-8")),
                io.BytesIO(b"installer bytes do not match"),
            ],
        ), patch("app.main.subprocess.Popen") as launch:
            response = self.client.post("/api/updates/install")

        self.assertEqual(response.status_code, 422)
        self.assertEqual(list(database.STORAGE.glob("assistant-*.sqlite3")), [])
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
