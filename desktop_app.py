from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import tkinter
from pathlib import Path
from tkinter import messagebox

import uvicorn
import webview

APP_NAME = "Clinthinker©"
WEBVIEW2_RUNTIME_KEY = (
    r"SOFTWARE\Microsoft\EdgeUpdate\Clients"
    r"\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
)
WEBVIEW2_INSTALLER = (
    "MicrosoftEdgeWebView2RuntimeInstallerX64.exe"
)


def project_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def show_error(message: str) -> None:
    try:
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror(APP_NAME, message, parent=root)
        root.destroy()
    except tkinter.TclError:
        print(message, file=sys.stderr)


def has_webview2_runtime() -> bool:
    if os.name != "nt":
        return True

    import winreg

    locations = (
        (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_32KEY),
        (winreg.HKEY_CURRENT_USER, 0),
    )
    for hive, view in locations:
        try:
            with winreg.OpenKey(
                hive,
                WEBVIEW2_RUNTIME_KEY,
                0,
                winreg.KEY_READ | view,
            ) as runtime_key:
                version, _ = winreg.QueryValueEx(runtime_key, "pv")
        except FileNotFoundError:
            continue
        parts = tuple(int(part) for part in str(version).split("."))
        if parts and any(parts):
            return True
    return False


def install_webview2_runtime(root: Path) -> bool:
    if has_webview2_runtime():
        return True

    installer = root / "Prerequisites" / WEBVIEW2_INSTALLER
    if not installer.is_file():
        raise FileNotFoundError(
            "The Microsoft WebView2 offline installer is missing. "
            "Keep the app's Prerequisites folder beside the executable."
        )

    dialog = tkinter.Tk()
    dialog.withdraw()
    try:
        install_now = messagebox.askyesno(
            APP_NAME,
            "Microsoft Edge WebView2 Runtime is required to display the app. "
            "Install it now for this Windows user using the included offline "
            "installer?",
            parent=dialog,
        )
    finally:
        dialog.destroy()
    if not install_now:
        return False

    completed = subprocess.run(
        [str(installer), "/silent", "/install"],
        check=False,
        timeout=600,
    )
    if completed.returncode != 0 and not has_webview2_runtime():
        raise RuntimeError(
            "The WebView2 Runtime installer failed "
            f"(exit code {completed.returncode})."
        )
    if not has_webview2_runtime():
        raise RuntimeError(
            "The WebView2 Runtime installer finished, but Windows still does "
            "not report the runtime as installed."
        )
    return True


def main() -> None:
    root = project_path()
    if not install_webview2_runtime(root):
        return

    if getattr(sys, "frozen", False):
        resource_root = Path(sys._MEIPASS)
        web_dir = resource_root / "frontend_dist"
    else:
        web_dir = root / "frontend" / "dist"
        sys.path.insert(0, str(root / "backend"))

    if not web_dir.is_dir():
        raise FileNotFoundError(
            f"The built interface was not found at {web_dir}. "
            "Run 'npm install && npm run build' in frontend before launching the desktop app."
        )

    data_root = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    storage_dir = data_root / "Topic Prediction Assistant" / "storage"
    renamed_storage = data_root / APP_NAME / "storage"
    if not renamed_storage.exists() and storage_dir.exists():
        renamed_storage.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(storage_dir), str(renamed_storage))
        try:
            storage_dir.parent.rmdir()
        except OSError:
            pass
        storage_dir = renamed_storage
    elif renamed_storage.exists():
        storage_dir = renamed_storage
    storage_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TOPIC_ASSISTANT_DATA_DIR"] = str(storage_dir)
    os.environ["TOPIC_ASSISTANT_WEB_DIR"] = str(web_dir)

    from app.main import app

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            access_log=False,
        )
    )
    server_thread = threading.Thread(
        target=server.run,
        name="topic-assistant-local-api",
        daemon=True,
    )
    server_thread.start()
    for _ in range(200):
        if server.started:
            break
        if not server_thread.is_alive():
            raise RuntimeError("The local prediction service stopped during startup.")
        time.sleep(0.1)
    else:
        server.should_exit = True
        raise TimeoutError("The local prediction service did not start in time.")

    try:
        webview.create_window(
            APP_NAME,
            f"http://127.0.0.1:{port}/",
            width=1120,
            height=820,
            min_size=(780, 640),
            background_color="#100d0a",
        )
        webview.start()
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        show_error(f"The app could not start:\n\n{error}")
        raise
