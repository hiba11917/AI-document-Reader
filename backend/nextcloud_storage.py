import os
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


NEXTCLOUD_BASE_URL = os.getenv("NEXTCLOUD_BASE_URL", "").strip().rstrip("/")
NEXTCLOUD_USERNAME = os.getenv("NEXTCLOUD_USERNAME", "").strip()
NEXTCLOUD_APP_PASSWORD = os.getenv("NEXTCLOUD_APP_PASSWORD", "").strip()
NEXTCLOUD_ROOT_FOLDER = os.getenv("NEXTCLOUD_ROOT_FOLDER", "AI Document Reader").strip("/")
REQUEST_TIMEOUT = int(os.getenv("NEXTCLOUD_REQUEST_TIMEOUT", "120"))


def is_nextcloud_enabled():
    # Only switch to remote storage when the full credential set is present in the backend environment.
    return bool(NEXTCLOUD_BASE_URL and NEXTCLOUD_USERNAME and NEXTCLOUD_APP_PASSWORD)


def _auth():
    return (NEXTCLOUD_USERNAME, NEXTCLOUD_APP_PASSWORD)


def _dav_base():
    # Nextcloud file operations go through WebDAV, so every upload/delete is built from the same base URL.
    return f"{NEXTCLOUD_BASE_URL}/remote.php/dav/files/{quote(NEXTCLOUD_USERNAME)}"


def _encode_segment(value):
    return quote(value, safe="")


def _remote_url_for_parts(*parts):
    encoded_parts = [_encode_segment(part.strip("/")) for part in parts if part and part.strip("/")]
    return f"{_dav_base()}/{'/'.join(encoded_parts)}"


def _ensure_remote_folder(folder_name):
    # Create the app folder on demand so first-time uploads do not require manual setup in Nextcloud.
    folder_url = _remote_url_for_parts(folder_name)
    response = requests.request("MKCOL", folder_url, auth=_auth(), timeout=REQUEST_TIMEOUT)
    if response.status_code not in (201, 301, 405):
        raise RuntimeError(
            f"Nextcloud folder creation failed with status {response.status_code}: {response.text}"
        )
    return folder_url


def upload_file(local_path, remote_filename):
    if not is_nextcloud_enabled():
        raise RuntimeError("Nextcloud is not configured.")

    # Upload the original file bytes to Nextcloud and return the remote URL stored in app metadata.
    _ensure_remote_folder(NEXTCLOUD_ROOT_FOLDER)
    destination_url = _remote_url_for_parts(NEXTCLOUD_ROOT_FOLDER, remote_filename)

    with open(local_path, "rb") as handle:
        response = requests.put(
            destination_url,
            data=handle,
            auth=_auth(),
            timeout=REQUEST_TIMEOUT,
        )

    if response.status_code not in (200, 201, 204):
        raise RuntimeError(
            f"Nextcloud upload failed with status {response.status_code}: {response.text}"
        )

    return destination_url


def delete_file(remote_url):
    # Treat missing remote files as already-cleaned-up so repeated deletes stay safe.
    if not is_nextcloud_enabled():
        return False

    response = requests.delete(remote_url, auth=_auth(), timeout=REQUEST_TIMEOUT)
    if response.status_code in (200, 204, 404):
        return True

    raise RuntimeError(
        f"Nextcloud delete failed with status {response.status_code}: {response.text}"
    )


def storage_mode_label():
    return "nextcloud" if is_nextcloud_enabled() else "local"
