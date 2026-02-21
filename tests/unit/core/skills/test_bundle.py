# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for skill bundle parsing and validation."""

import zipfile
from io import BytesIO

import pytest

from llama_stack.core.skills.bundle import (
    MAX_FILE_COUNT,
    MAX_UNCOMPRESSED_SIZE,
    BundleError,
    extract_metadata,
    extract_zip,
    files_to_map,
    find_manifest,
    parse_frontmatter,
    validate_bundle,
)

# --- parse_frontmatter tests ---


class TestParseFrontmatter:
    """Tests for YAML frontmatter extraction."""

    def test_valid_frontmatter(self):
        content = "---\nname: csv-insights\ndescription: Analyze CSV files.\n---\n# Body"
        result = parse_frontmatter(content)
        assert result["name"] == "csv-insights"
        assert result["description"] == "Analyze CSV files."

    def test_name_only(self):
        content = "---\nname: my-skill\n---\n# Body"
        result = parse_frontmatter(content)
        assert result["name"] == "my-skill"
        assert "description" not in result

    def test_no_frontmatter(self):
        content = "# Just a heading\nSome text."
        result = parse_frontmatter(content)
        assert result == {}

    def test_incomplete_frontmatter(self):
        content = "---\nname: broken"
        result = parse_frontmatter(content)
        assert result == {}

    def test_empty_content(self):
        result = parse_frontmatter("")
        assert result == {}

    def test_frontmatter_with_leading_whitespace(self):
        content = "\n\n---\nname: my-skill\n---\n"
        result = parse_frontmatter(content)
        assert result["name"] == "my-skill"

    def test_invalid_yaml(self):
        content = "---\n: invalid: yaml: [broken\n---\n"
        result = parse_frontmatter(content)
        assert result == {}

    def test_non_dict_yaml(self):
        content = "---\n- just a list\n---\n"
        result = parse_frontmatter(content)
        assert result == {}

    def test_numeric_name_coerced_to_string(self):
        content = "---\nname: 42\n---\n"
        result = parse_frontmatter(content)
        assert result["name"] == "42"


# --- find_manifest tests ---


class TestFindManifest:
    """Tests for manifest file discovery."""

    def test_finds_uppercase(self):
        file_map = {"SKILL.md": b"content", "run.sh": b"echo hi"}
        path, data = find_manifest(file_map)
        assert path == "SKILL.md"

    def test_finds_lowercase(self):
        file_map = {"skill.md": b"content", "run.sh": b"echo hi"}
        path, data = find_manifest(file_map)
        assert path == "skill.md"

    def test_finds_nested(self):
        file_map = {"my_skill/SKILL.md": b"content", "my_skill/run.sh": b"echo hi"}
        path, data = find_manifest(file_map)
        assert path == "my_skill/SKILL.md"

    def test_no_manifest_raises(self):
        file_map = {"run.sh": b"echo hi", "README.md": b"docs"}
        with pytest.raises(BundleError, match="must contain a SKILL.md"):
            find_manifest(file_map)

    def test_multiple_manifests_raises(self):
        file_map = {"SKILL.md": b"a", "sub/skill.md": b"b"}
        with pytest.raises(BundleError, match="exactly one"):
            find_manifest(file_map)


# --- validate_bundle tests ---


class TestValidateBundle:
    """Tests for bundle size and count validation."""

    def test_valid_bundle(self):
        file_map = {"SKILL.md": b"content", "run.py": b"print('hello')"}
        validate_bundle(file_map)  # should not raise

    def test_too_many_files(self):
        file_map = {f"file_{i}.txt": b"data" for i in range(MAX_FILE_COUNT + 1)}
        with pytest.raises(BundleError, match="file count"):
            validate_bundle(file_map)

    def test_file_too_large(self):
        file_map = {"big.bin": b"x" * (MAX_UNCOMPRESSED_SIZE + 1)}
        with pytest.raises(BundleError, match="maximum size"):
            validate_bundle(file_map)


# --- extract_zip tests ---


class TestExtractZip:
    """Tests for zip archive extraction."""

    def _make_zip(self, file_map: dict[str, bytes]) -> bytes:
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, data in file_map.items():
                zf.writestr(name, data)
        return buf.getvalue()

    def test_basic_extraction(self):
        data = self._make_zip({"SKILL.md": b"# Skill", "run.py": b"print(1)"})
        result = extract_zip(data)
        assert "SKILL.md" in result
        assert "run.py" in result
        assert result["SKILL.md"] == b"# Skill"

    def test_skips_directories(self):
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("mydir/", "")
            zf.writestr("mydir/file.txt", "content")
        result = extract_zip(buf.getvalue())
        assert "mydir/" not in result
        assert "mydir/file.txt" in result

    def test_skips_macosx_metadata(self):
        data = self._make_zip(
            {
                "__MACOSX/._SKILL.md": b"meta",
                "SKILL.md": b"# Skill",
            }
        )
        result = extract_zip(data)
        assert "__MACOSX/._SKILL.md" not in result
        assert "SKILL.md" in result

    def test_skips_hidden_files(self):
        data = self._make_zip(
            {
                ".DS_Store": b"meta",
                "SKILL.md": b"# Skill",
            }
        )
        result = extract_zip(data)
        assert ".DS_Store" not in result
        assert "SKILL.md" in result

    def test_invalid_zip_raises(self):
        with pytest.raises(BundleError, match="not a valid zip"):
            extract_zip(b"this is not a zip file")

    def test_oversized_zip_raises(self):
        from llama_stack.core.skills.bundle import MAX_ZIP_SIZE

        with pytest.raises(BundleError, match="maximum size"):
            extract_zip(b"x" * (MAX_ZIP_SIZE + 1))


# --- files_to_map tests ---


class TestFilesToMap:
    """Tests for multipart file conversion."""

    def test_basic_conversion(self):
        result = files_to_map(["SKILL.md", "run.sh"], [b"skill", b"bash"])
        assert result == {"SKILL.md": b"skill", "run.sh": b"bash"}

    def test_strips_directory_components(self):
        result = files_to_map(["../../etc/SKILL.md"], [b"content"])
        assert "SKILL.md" in result
        assert "../../etc/SKILL.md" not in result

    def test_empty_filename_gets_fallback(self):
        result = files_to_map(["", "SKILL.md"], [b"a", b"b"])
        assert len(result) == 2
        assert "SKILL.md" in result


# --- extract_metadata tests ---


class TestExtractMetadata:
    """Tests for metadata extraction from SKILL.md frontmatter."""

    def test_extracts_name_and_description(self):
        file_map = {
            "SKILL.md": b"---\nname: csv-insights\ndescription: Analyze CSVs.\n---\n# Body",
            "run.py": b"print(1)",
        }
        name, desc = extract_metadata(file_map)
        assert name == "csv-insights"
        assert desc == "Analyze CSVs."

    def test_missing_description_defaults_to_empty(self):
        file_map = {"SKILL.md": b"---\nname: my-skill\n---\n"}
        name, desc = extract_metadata(file_map)
        assert name == "my-skill"
        assert desc == ""

    def test_missing_name_falls_back_to_filename(self):
        file_map = {"my_cool_skill/SKILL.md": b"---\ndescription: A skill\n---\n"}
        name, desc = extract_metadata(file_map)
        assert name == "untitled"
        assert desc == "A skill"

    def test_no_frontmatter_uses_defaults(self):
        file_map = {"SKILL.md": b"# Just a heading\nNo frontmatter here."}
        name, desc = extract_metadata(file_map)
        assert name == "untitled"
        assert desc == ""

    def test_no_manifest_raises(self):
        file_map = {"README.md": b"# Not a skill"}
        with pytest.raises(BundleError):
            extract_metadata(file_map)
