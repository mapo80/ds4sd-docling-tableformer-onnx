#!/usr/bin/env python3
"""Download all assets from a GitHub release."""
import argparse
import os
import sys
import pathlib
import urllib.request
import urllib.error
import json
from typing import Optional

API_URL_TEMPLATE = "https://api.github.com/repos/{repo}/releases/tags/{tag}"


def read_token(env_var: Optional[str] = None) -> str:
    candidates = [
        env_var,
        os.environ.get("GITHUB_TOKEN"),
        os.environ.get("GITHUB_PAT"),
        os.environ.get("GH_TOKEN"),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    raise SystemExit("GitHub token not provided. Set GITHUB_TOKEN or GH_TOKEN environment variable.")


def request_json(url: str, token: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "release-downloader",
        },
    )
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="ignore")
        raise SystemExit(f"Failed to query GitHub API ({exc.code}): {payload}") from exc


def download_asset(url: str, dest: pathlib.Path, token: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/octet-stream",
            "User-Agent": "release-downloader",
        },
    )
    try:
        with urllib.request.urlopen(req) as response, dest.open("wb") as fh:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="ignore")
        raise SystemExit(f"Failed to download {url} ({exc.code}): {payload}") from exc


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Download GitHub release assets")
    parser.add_argument("--repo", required=True, help="Repository in <owner>/<name> format")
    parser.add_argument("--tag", required=True, help="Release tag (e.g. v1.0.0)")
    parser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
        help="Destination directory for downloaded assets",
    )
    parser.add_argument(
        "--token",
        help="GitHub token (overrides environment variables)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip assets that already exist with the expected size",
    )
    args = parser.parse_args(argv)

    token = read_token(args.token)
    release_url = API_URL_TEMPLATE.format(repo=args.repo, tag=args.tag)
    release = request_json(release_url, token)

    assets = release.get("assets", [])
    if not assets:
        print("No assets found in release", file=sys.stderr)
        return 0

    args.output.mkdir(parents=True, exist_ok=True)

    for asset in assets:
        name = asset.get("name")
        download_url = asset.get("browser_download_url")
        size = asset.get("size")
        if not name or not download_url:
            continue

        destination = args.output / name
        if args.skip_existing and destination.exists() and size is not None:
            existing_size = destination.stat().st_size
            if existing_size == size:
                print(f"Skipping {name} (already downloaded)")
                continue

        print(f"Downloading {name} -> {destination}")
        download_asset(download_url, destination, token)

        if size is not None:
            downloaded_size = destination.stat().st_size
            if downloaded_size != size:
                raise SystemExit(
                    f"Size mismatch for {name}: expected {size} bytes, got {downloaded_size} bytes"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
