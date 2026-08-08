"""Validate local course links and optionally make real HTTP requests.

Usage:
    python tools/validate_course.py
    python tools/validate_course.py --external
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
COURSE_SUFFIXES = {".md", ".html"}
LINK_PATTERN = re.compile(r'(?:href=["\']|\]\()(?P<url>[^"\')#\s]+)')


def collect_links() -> tuple[list[tuple[Path, str]], set[str]]:
    local_links: list[tuple[Path, str]] = []
    external_links: set[str] = set()

    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in COURSE_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        for match in LINK_PATTERN.finditer(text):
            url = match.group("url")
            if url.startswith(("http://", "https://")):
                external_links.add(url)
            elif not url.startswith(("mailto:", "javascript:", "data:")):
                local_links.append((path, url))

    return local_links, external_links


def validate_local(links: list[tuple[Path, str]]) -> list[str]:
    failures: list[str] = []
    for source, raw_link in links:
        link_path = unquote(urlsplit(raw_link).path)
        target = (source.parent / link_path).resolve()
        if not target.exists():
            failures.append(f"{source.relative_to(ROOT)} -> {raw_link}")
    return failures


def validate_external(urls: set[str]) -> list[str]:
    failures: list[str] = []
    headers = {"User-Agent": "Mozilla/5.0 course-link-validator/1.0"}
    for url in sorted(urls):
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                code = response.status
            print(f"HTTP {code} {url}")
            if not 200 <= code < 400:
                failures.append(f"HTTP {code} {url}")
        except (urllib.error.URLError, TimeoutError) as error:
            failures.append(f"HTTP ERROR {url}: {error}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external",
        action="store_true",
        help="also issue real GET requests for every external URL",
    )
    args = parser.parse_args()

    local_links, external_links = collect_links()
    failures = validate_local(local_links)
    print(f"Local links checked: {len(local_links)}")

    if args.external:
        failures.extend(validate_external(external_links))
        print(f"External links checked: {len(external_links)}")

    if failures:
        print("\nValidation failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("Course link validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
