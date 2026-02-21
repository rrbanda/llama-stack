# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Skill bundle parsing and validation.

Handles SKILL.md frontmatter extraction, bundle structure validation,
and zip archive processing for skill uploads.
"""

import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath

import yaml

from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="skills")

MANIFEST_NAMES = {"skill.md"}

MAX_FILE_COUNT = 500
MAX_UNCOMPRESSED_SIZE = 25 * 1024 * 1024  # 25 MB per file
MAX_ZIP_SIZE = 50 * 1024 * 1024  # 50 MB


class BundleError(ValueError):
    """Raised when a skill bundle fails validation."""


def parse_frontmatter(content: str) -> dict[str, str]:
    """Extract YAML frontmatter from a SKILL.md file.

    Frontmatter is delimited by ``---`` markers at the start of the file::

        ---
        name: my-skill
        description: Does something useful.
        ---
        # Rest of document

    Returns a dict with the parsed fields, or an empty dict if no
    valid frontmatter is found.
    """
    stripped = content.lstrip()
    if not stripped.startswith("---"):
        return {}

    parts = stripped.split("---", 2)
    if len(parts) < 3:
        return {}

    try:
        data = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        logger.warning("Failed to parse SKILL.md frontmatter as YAML")
        return {}

    if not isinstance(data, dict):
        return {}

    result: dict[str, str] = {}
    if "name" in data:
        result["name"] = str(data["name"])
    if "description" in data:
        result["description"] = str(data["description"])
    return result


def find_manifest(file_map: dict[str, bytes]) -> tuple[str, bytes]:
    """Locate the SKILL.md manifest in a file map.

    Matching is case-insensitive.  Exactly one manifest must be present.

    Returns (relative_path, content) of the manifest.
    """
    matches = [(path, data) for path, data in file_map.items() if PurePosixPath(path).name.lower() in MANIFEST_NAMES]

    if not matches:
        raise BundleError("Skill bundle must contain a SKILL.md manifest file")

    if len(matches) > 1:
        paths = ", ".join(m[0] for m in matches)
        raise BundleError(f"Skill bundle must contain exactly one SKILL.md manifest, found: {paths}")

    return matches[0]


def validate_bundle(file_map: dict[str, bytes]) -> None:
    """Validate a skill bundle against size and count limits."""
    if len(file_map) > MAX_FILE_COUNT:
        raise BundleError(f"Skill bundle exceeds maximum file count ({MAX_FILE_COUNT})")

    for path, data in file_map.items():
        if len(data) > MAX_UNCOMPRESSED_SIZE:
            size_mb = len(data) / (1024 * 1024)
            raise BundleError(
                f"File '{path}' exceeds maximum size of {MAX_UNCOMPRESSED_SIZE // (1024 * 1024)} MB ({size_mb:.1f} MB)"
            )


def extract_zip(data: bytes) -> dict[str, bytes]:
    """Extract a zip archive into a file map.

    Skips directories, hidden files (``__MACOSX``, ``.*``), and applies
    path sanitisation.  Returns ``{relative_path: content}`` pairs.
    """
    if len(data) > MAX_ZIP_SIZE:
        raise BundleError(f"Zip file exceeds maximum size of {MAX_ZIP_SIZE // (1024 * 1024)} MB")

    buf = BytesIO(data)
    if not zipfile.is_zipfile(buf):
        raise BundleError("Uploaded file is not a valid zip archive")

    buf.seek(0)
    file_map: dict[str, bytes] = {}
    with zipfile.ZipFile(buf, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            parts = PurePosixPath(info.filename).parts
            if any(p.startswith(".") or p == "__MACOSX" for p in parts):
                continue

            file_map[info.filename] = zf.read(info.filename)

    return file_map


def files_to_map(names: list[str], contents: list[bytes]) -> dict[str, bytes]:
    """Convert parallel lists of filenames and contents into a file map.

    Sanitises filenames by stripping directory components.
    """
    file_map: dict[str, bytes] = {}
    for name, content in zip(names, contents, strict=True):
        safe_name = Path(name).name if name else None
        if not safe_name:
            safe_name = f"file_{len(file_map)}"
        file_map[safe_name] = content
    return file_map


def extract_metadata(file_map: dict[str, bytes]) -> tuple[str, str]:
    """Extract name and description from the bundle's SKILL.md.

    Returns (name, description).  Falls back to the manifest filename
    (without extension) if ``name`` is missing from frontmatter, and
    to an empty string for ``description``.
    """
    manifest_path, manifest_content = find_manifest(file_map)

    frontmatter = parse_frontmatter(manifest_content.decode("utf-8", errors="replace"))
    name = frontmatter.get("name", "")
    description = frontmatter.get("description", "")

    if not name:
        stem = PurePosixPath(manifest_path).stem
        name = stem if stem.lower() != "skill" else "untitled"

    return name, description
