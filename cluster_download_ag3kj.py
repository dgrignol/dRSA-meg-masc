#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone OSF downloader tailored for the MEG-MASK project (ag3kj).

Copy this script to the cluster, then run:
    python3 cluster_download_ag3kj.py --dest /path/to/MEG-MASK

Re-running the same command resumes any interrupted downloads.
"""

import argparse
import os
import sys
import time
from typing import Dict, Generator, Iterable, List, Optional, Tuple


def ensure_requests():
    """Import requests, installing it locally if necessary."""
    try:
        import requests  # type: ignore
        return requests
    except ImportError:  # pragma: no cover - executed only when missing
        import subprocess

        print("requests not found; installing with pip --user …", file=sys.stderr)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--user", "requests"]
            )
        except subprocess.CalledProcessError as exc:
            print(
                "Failed to install requests automatically. "
                "Install it manually with 'python3 -m pip install --user requests'.",
                file=sys.stderr,
            )
            raise SystemExit(exc.returncode)
        import importlib

        return importlib.import_module("requests")


requests = ensure_requests()

API_BASE = "https://api.osf.io/v2"


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} {units[-1]}"


class OSFClient:
    def __init__(self, timeout: int = 60, token: Optional[str] = None):
        self.session = requests.Session()
        self.timeout = timeout
        self.session.headers.update(
            {
                "Accept": "application/vnd.api+json",
                "User-Agent": "osf-downloader/cluster",
            }
        )
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

    def get_json(self, url: str) -> Dict:
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def list_providers(self, node_id: str) -> List[Dict]:
        url = f"{API_BASE}/nodes/{node_id}/files/"
        data = self.get_json(url)
        return data.get("data", [])

    def provider_root_children_url(self, node_id: str, provider: str = "osfstorage") -> str:
        providers = self.list_providers(node_id)
        for entry in providers:
            attrs = entry.get("attributes", {})
            if attrs.get("name") == provider:
                rel = entry.get("relationships", {}).get("files", {})
                href = rel.get("links", {}).get("related", {}).get("href")
                if href:
                    return href
        raise RuntimeError(f"Provider {provider} not found for node {node_id}")

    def iter_collection(self, url: str) -> Generator[Dict, None, None]:
        next_url = url
        if "page[size]" not in next_url:
            sep = "&" if "?" in next_url else "?"
            next_url = f"{next_url}{sep}page[size]=100"
        while next_url:
            data = self.get_json(next_url)
            for item in data.get("data", []):
                yield item
            next_url = data.get("links", {}).get("next")

    def iter_folder_tree(self, folder_url: str) -> Generator[Dict, None, None]:
        for item in self.iter_collection(folder_url):
            kind = item.get("attributes", {}).get("kind")
            if kind == "file":
                yield item
            elif kind == "folder":
                rel = item.get("relationships", {}).get("files", {})
                href = rel.get("links", {}).get("related", {}).get("href")
                if href:
                    yield from self.iter_folder_tree(href)


def download_with_resume(
    url: str,
    local_path: str,
    headers: Optional[Dict[str, str]] = None,
    chunk_size: int = 1024 * 1024,
    expected_size: Optional[int] = None,
    timeout: int = 120,
) -> None:
    dir_name = os.path.dirname(local_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    existing_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
    req_headers = dict(headers) if headers else {}
    if existing_size > 0:
        req_headers["Range"] = f"bytes={existing_size}-"

    resp = requests.get(url, headers=req_headers, stream=True, timeout=timeout)

    if resp.status_code == 206:
        mode = "ab"
    elif resp.status_code == 200:
        mode = "wb"
        existing_size = 0
    else:
        raise RuntimeError(f"Unexpected status code {resp.status_code} for {url}")

    bytes_written = existing_size
    with open(local_path, mode) as fh:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            fh.write(chunk)
            bytes_written += len(chunk)

    if expected_size is not None and expected_size > 0:
        final_size = os.path.getsize(local_path)
        if final_size != expected_size:
            raise RuntimeError(
                f"Size mismatch for {local_path}: got {final_size}, expected {expected_size}"
            )


def collect_files(client: OSFClient, node_id: str) -> List[Tuple[str, int, str]]:
    files: List[Tuple[str, int, str]] = []
    root_children = client.provider_root_children_url(node_id)
    for item in client.iter_folder_tree(root_children):
        attrs = item.get("attributes", {})
        links = item.get("links", {})
        path = attrs.get("materialized_path") or attrs.get("path") or attrs.get("name")
        size = int(attrs.get("size") or 0)
        download_url = links.get("download")
        if not download_url:
            guid = attrs.get("guid")
            if guid:
                download_url = f"https://osf.io/download/{guid}/"
        if not (path and download_url):
            continue
        files.append((path, size, download_url))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the MEG-MASK OSF dataset (project ag3kj) with resume support."
    )
    parser.add_argument(
        "--project",
        default="ag3kj",
        help="OSF project/node id (default: ag3kj)",
    )
    parser.add_argument(
        "--dest",
        default="MEG-MASK",
        help="Destination directory for downloaded files",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds (default: 180)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per file (default: 5)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional OSF personal access token for private datasets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list files, don't download.",
    )
    args = parser.parse_args()

    client = OSFClient(timeout=args.timeout, token=args.token)

    print(f"Indexing files for OSF node {args.project}…", flush=True)
    try:
        files = collect_files(client, args.project)
    except Exception as exc:
        print(f"Failed to list files: {exc}", file=sys.stderr)
        return 1

    if not files:
        print("No files discovered; nothing to do.")
        return 0

    total_bytes = sum(size for _, size, _ in files)
    print(f"Found {len(files)} files, total size ≈ {human_size(total_bytes)}")

    if args.dry_run:
        for path, size, _ in files[:50]:
            print(f"- {path} ({human_size(size)})")
        if len(files) > 50:
            print(f"… and {len(files) - 50} more")
        return 0

    base_dir = os.path.abspath(args.dest)
    os.makedirs(base_dir, exist_ok=True)

    failures: List[str] = []
    for idx, (materialized_path, size, url) in enumerate(files, start=1):
        rel_path = materialized_path.lstrip("/")
        local_path = os.path.join(base_dir, rel_path)

        if os.path.exists(local_path) and size > 0:
            try:
                if os.path.getsize(local_path) == size:
                    print(f"[{idx}/{len(files)}] Skip (already complete): {rel_path}")
                    continue
            except OSError:
                pass

        print(
            f"[{idx}/{len(files)}] Downloading {rel_path} ({human_size(size)})",
            flush=True,
        )

        for attempt in range(1, args.max_retries + 1):
            try:
                download_with_resume(
                    url,
                    local_path,
                    expected_size=size if size > 0 else None,
                    timeout=args.timeout,
                )
                break
            except KeyboardInterrupt:
                print("\nInterrupted by user; re-run the script to resume.", file=sys.stderr)
                return 130
            except Exception as exc:
                wait = min(60, 2 ** attempt)
                print(
                    f"  attempt {attempt}/{args.max_retries} failed: {exc} — retrying in {wait}s",
                    flush=True,
                )
                time.sleep(wait)
        else:
            print(f"  FAILED: {rel_path}")
            failures.append(rel_path)

    if failures:
        print("\nCompleted with failures:")
        for path in failures:
            print(f"- {path}")
        return 2

    print("All files downloaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

