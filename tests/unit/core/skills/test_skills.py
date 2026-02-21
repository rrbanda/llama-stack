# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the Skills API implementation."""

import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from zipfile import ZipFile

import pytest

from llama_stack.core.skills.skills import (
    KEY_PREFIX,
    VERSION_KEY_PREFIX,
    SkillServiceImpl,
)
from llama_stack_api import (
    CreateSkillVersionRequest,
    DeleteSkillRequest,
    GetSkillContentRequest,
    GetSkillRequest,
    GetSkillVersionRequest,
    ListSkillsRequest,
    ListSkillVersionsRequest,
    Order,
    Skill,
    SkillNotFoundError,
    SkillVersion,
    UpdateSkillRequest,
)

# --- Fixtures ---


@pytest.fixture
def mock_kvstore():
    """Create a mock KVStore with in-memory storage."""
    storage = {}

    class MockKVStore:
        async def set(self, key, value):
            storage[key] = value

        async def get(self, key):
            return storage.get(key)

        async def delete(self, key):
            del storage[key]

        async def keys_in_range(self, start_key, end_key):
            return [k for k in storage.keys() if start_key <= k < end_key]

        async def close(self):
            pass

        async def shutdown(self):
            pass

        @property
        def _storage(self):
            return storage

    return MockKVStore()


@pytest.fixture
def storage_dir():
    """Create a temporary directory for skill file storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
async def skill_service(mock_kvstore, storage_dir):
    """Create a SkillServiceImpl with mocked dependencies."""
    mock_config = MagicMock()
    service = SkillServiceImpl(mock_config)
    service.kvstore = mock_kvstore
    service.storage_dir = Path(storage_dir)
    return service


SAMPLE_MANIFEST = b"---\nname: test-skill\ndescription: A test skill.\n---\n# Test Skill\n"
SAMPLE_MANIFEST_V2 = b"---\nname: test-skill-v2\ndescription: Updated skill.\n---\n# v2\n"


def _make_upload_file(filename: str, content: bytes, content_type: str | None = None) -> MagicMock:
    """Create a mock UploadFile."""
    upload_file = AsyncMock()
    upload_file.filename = filename
    upload_file.content_type = content_type
    upload_file.read = AsyncMock(return_value=content)
    return upload_file


# --- create_skill tests ---


class TestCreateSkill:
    """Tests for create_skill method."""

    async def test_create_skill_returns_skill(self, skill_service):
        files = [_make_upload_file("SKILL.md", SAMPLE_MANIFEST)]
        result = await skill_service.create_skill(files)

        assert isinstance(result, Skill)
        assert result.id.startswith("skill_")
        assert result.name == "test-skill"
        assert result.description == "A test skill."
        assert result.default_version == "1"
        assert result.latest_version == "1"
        assert result.object == "skill"

    async def test_create_skill_stores_in_kvstore(self, skill_service, mock_kvstore):
        files = [_make_upload_file("SKILL.md", SAMPLE_MANIFEST)]
        result = await skill_service.create_skill(files)

        assert await mock_kvstore.get(f"{KEY_PREFIX}{result.id}") is not None

    async def test_create_skill_creates_version_1(self, skill_service, mock_kvstore):
        """Creating a skill also creates version 1 metadata."""
        files = [_make_upload_file("SKILL.md", SAMPLE_MANIFEST)]
        result = await skill_service.create_skill(files)

        ver_json = await mock_kvstore.get(f"{VERSION_KEY_PREFIX}{result.id}:1")
        assert ver_json is not None

    async def test_create_skill_stores_files_in_version_dir(self, skill_service, storage_dir):
        """Files are stored under {skill_id}/v1/."""
        files = [
            _make_upload_file("SKILL.md", SAMPLE_MANIFEST),
            _make_upload_file("run.sh", b"#!/bin/bash\necho hello"),
        ]
        result = await skill_service.create_skill(files)

        v1_dir = Path(storage_dir) / result.id / "v1"
        assert v1_dir.exists()
        assert (v1_dir / "SKILL.md").exists()
        assert (v1_dir / "run.sh").exists()

    async def test_create_skill_extracts_frontmatter(self, skill_service):
        manifest = b"---\nname: csv-insights\ndescription: Analyze CSV data.\n---\n# CSV"
        files = [_make_upload_file("SKILL.md", manifest)]
        result = await skill_service.create_skill(files)
        assert result.name == "csv-insights"
        assert result.description == "Analyze CSV data."

    async def test_create_skill_no_files_raises(self, skill_service):
        with pytest.raises(ValueError, match="At least one file"):
            await skill_service.create_skill([])

    async def test_create_skill_no_manifest_raises(self, skill_service):
        from llama_stack.core.skills.bundle import BundleError

        files = [_make_upload_file("README.md", b"# Not a skill")]
        with pytest.raises(BundleError, match="SKILL.md"):
            await skill_service.create_skill(files)

    async def test_create_skill_from_zip(self, skill_service):
        import zipfile

        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("my_skill/SKILL.md", SAMPLE_MANIFEST.decode())
            zf.writestr("my_skill/run.py", "print('hello')")

        files = [_make_upload_file("my_skill.zip", buf.getvalue(), content_type="application/zip")]
        result = await skill_service.create_skill(files)
        assert result.name == "test-skill"


# --- list_skills tests ---


class TestListSkills:
    """Tests for list_skills method."""

    async def test_list_skills_empty(self, skill_service):
        result = await skill_service.list_skills(ListSkillsRequest())
        assert result.data == []
        assert result.has_more is False

    async def test_list_skills_returns_all(self, skill_service):
        for _ in range(3):
            await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        result = await skill_service.list_skills(ListSkillsRequest())
        assert len(result.data) == 3

    async def test_list_skills_default_order_is_desc(self, skill_service):
        for _ in range(2):
            await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        result = await skill_service.list_skills(ListSkillsRequest())
        assert result.data[0].created_at >= result.data[-1].created_at

    async def test_list_skills_with_limit(self, skill_service):
        for _ in range(5):
            await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        result = await skill_service.list_skills(ListSkillsRequest(limit=2))
        assert len(result.data) == 2
        assert result.has_more is True

    async def test_list_skills_with_after(self, skill_service):
        for _ in range(3):
            await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        first = await skill_service.list_skills(ListSkillsRequest(limit=1))
        second = await skill_service.list_skills(ListSkillsRequest(after=first.last_id))
        assert second.data[0].id != first.data[0].id

    async def test_list_skills_asc_order(self, skill_service):
        for _ in range(2):
            await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        result = await skill_service.list_skills(ListSkillsRequest(order=Order.asc))
        assert result.data[0].created_at <= result.data[-1].created_at


# --- get_skill tests ---


class TestGetSkill:
    """Tests for get_skill method."""

    async def test_get_skill_returns_skill(self, skill_service):
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        result = await skill_service.get_skill(GetSkillRequest(skill_id=created.id))
        assert result.id == created.id

    async def test_get_skill_not_found(self, skill_service):
        with pytest.raises(SkillNotFoundError):
            await skill_service.get_skill(GetSkillRequest(skill_id="nonexistent"))


# --- update_skill tests ---


class TestUpdateSkill:
    """Tests for update_skill method."""

    async def test_update_default_version(self, skill_service):
        """Updating default_version to an existing version succeeds."""
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        await skill_service.create_skill_version(
            CreateSkillVersionRequest(skill_id=created.id),
            [_make_upload_file("SKILL.md", SAMPLE_MANIFEST_V2)],
        )

        result = await skill_service.update_skill(UpdateSkillRequest(skill_id=created.id, default_version="2"))
        assert result.default_version == "2"

    async def test_update_to_nonexistent_version_raises(self, skill_service):
        """Updating default_version to a version that doesn't exist raises ValueError."""
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        with pytest.raises(ValueError, match="does not exist"):
            await skill_service.update_skill(UpdateSkillRequest(skill_id=created.id, default_version="99"))

    async def test_update_persists_change(self, skill_service):
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        await skill_service.create_skill_version(
            CreateSkillVersionRequest(skill_id=created.id),
            [_make_upload_file("SKILL.md", SAMPLE_MANIFEST_V2)],
        )
        await skill_service.update_skill(UpdateSkillRequest(skill_id=created.id, default_version="2"))
        fetched = await skill_service.get_skill(GetSkillRequest(skill_id=created.id))
        assert fetched.default_version == "2"

    async def test_update_nonexistent_skill_raises(self, skill_service):
        with pytest.raises(SkillNotFoundError):
            await skill_service.update_skill(UpdateSkillRequest(skill_id="nonexistent", default_version="1"))


# --- delete_skill tests ---


class TestDeleteSkill:
    """Tests for delete_skill method."""

    async def test_delete_skill_returns_deleted(self, skill_service):
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        result = await skill_service.delete_skill(DeleteSkillRequest(skill_id=created.id))
        assert result.id == created.id
        assert result.deleted is True
        assert result.object == "skill.deleted"

    async def test_delete_skill_removes_from_kvstore(self, skill_service, mock_kvstore):
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        await skill_service.delete_skill(DeleteSkillRequest(skill_id=created.id))
        assert await mock_kvstore.get(f"{KEY_PREFIX}{created.id}") is None

    async def test_delete_skill_removes_version_metadata(self, skill_service, mock_kvstore):
        """Deleting a skill also removes all version keys."""
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        await skill_service.create_skill_version(
            CreateSkillVersionRequest(skill_id=created.id),
            [_make_upload_file("SKILL.md", SAMPLE_MANIFEST_V2)],
        )
        await skill_service.delete_skill(DeleteSkillRequest(skill_id=created.id))
        assert await mock_kvstore.get(f"{VERSION_KEY_PREFIX}{created.id}:1") is None
        assert await mock_kvstore.get(f"{VERSION_KEY_PREFIX}{created.id}:2") is None

    async def test_delete_skill_removes_files(self, skill_service, storage_dir):
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        skill_dir = Path(storage_dir) / created.id
        assert skill_dir.exists()
        await skill_service.delete_skill(DeleteSkillRequest(skill_id=created.id))
        assert not skill_dir.exists()

    async def test_delete_nonexistent_skill_raises(self, skill_service):
        with pytest.raises(SkillNotFoundError):
            await skill_service.delete_skill(DeleteSkillRequest(skill_id="nonexistent"))

    async def test_delete_skill_not_in_list(self, skill_service):
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        await skill_service.delete_skill(DeleteSkillRequest(skill_id=created.id))
        result = await skill_service.list_skills(ListSkillsRequest())
        assert len(result.data) == 0


# --- get_skill_content tests ---


class TestGetSkillContent:
    """Tests for get_skill_content method."""

    async def test_get_content_returns_zip(self, skill_service):
        files = [
            _make_upload_file("SKILL.md", SAMPLE_MANIFEST),
            _make_upload_file("run.sh", b"#!/bin/bash\necho hello"),
        ]
        created = await skill_service.create_skill(files)

        response = await skill_service.get_skill_content(GetSkillContentRequest(skill_id=created.id))
        assert response.media_type == "application/zip"
        zf = ZipFile(BytesIO(response.body))
        assert "SKILL.md" in zf.namelist()
        assert "run.sh" in zf.namelist()

    async def test_get_content_specific_version(self, skill_service):
        """Requesting a specific version returns that version's content."""
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        await skill_service.create_skill_version(
            CreateSkillVersionRequest(skill_id=created.id),
            [_make_upload_file("SKILL.md", SAMPLE_MANIFEST_V2), _make_upload_file("v2.txt", b"v2")],
        )

        resp_v1 = await skill_service.get_skill_content(GetSkillContentRequest(skill_id=created.id, version=1))
        resp_v2 = await skill_service.get_skill_content(GetSkillContentRequest(skill_id=created.id, version=2))

        zf1 = ZipFile(BytesIO(resp_v1.body))
        zf2 = ZipFile(BytesIO(resp_v2.body))
        assert "v2.txt" not in zf1.namelist()
        assert "v2.txt" in zf2.namelist()

    async def test_get_content_nonexistent_skill_raises(self, skill_service):
        with pytest.raises(SkillNotFoundError):
            await skill_service.get_skill_content(GetSkillContentRequest(skill_id="nonexistent"))


# --- Versioning tests ---


class TestCreateSkillVersion:
    """Tests for create_skill_version method."""

    async def test_create_version_returns_skill_version(self, skill_service):
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])

        ver = await skill_service.create_skill_version(
            CreateSkillVersionRequest(skill_id=created.id),
            [_make_upload_file("SKILL.md", SAMPLE_MANIFEST_V2)],
        )
        assert isinstance(ver, SkillVersion)
        assert ver.version == 2
        assert ver.skill_id == created.id
        assert ver.name == "test-skill-v2"
        assert ver.object == "skill.version"

    async def test_create_version_increments_latest(self, skill_service):
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])

        await skill_service.create_skill_version(
            CreateSkillVersionRequest(skill_id=created.id),
            [_make_upload_file("SKILL.md", SAMPLE_MANIFEST_V2)],
        )

        skill = await skill_service.get_skill(GetSkillRequest(skill_id=created.id))
        assert skill.latest_version == "2"
        assert skill.default_version == "1"

    async def test_create_version_stores_files_in_version_dir(self, skill_service, storage_dir):
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])

        await skill_service.create_skill_version(
            CreateSkillVersionRequest(skill_id=created.id),
            [_make_upload_file("SKILL.md", SAMPLE_MANIFEST_V2), _make_upload_file("new.txt", b"new")],
        )

        v2_dir = Path(storage_dir) / created.id / "v2"
        assert v2_dir.exists()
        assert (v2_dir / "SKILL.md").exists()
        assert (v2_dir / "new.txt").exists()

    async def test_create_version_updates_skill_metadata(self, skill_service):
        """Creating a new version updates the parent skill's name and description."""
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        assert created.name == "test-skill"

        await skill_service.create_skill_version(
            CreateSkillVersionRequest(skill_id=created.id),
            [_make_upload_file("SKILL.md", SAMPLE_MANIFEST_V2)],
        )

        skill = await skill_service.get_skill(GetSkillRequest(skill_id=created.id))
        assert skill.name == "test-skill-v2"
        assert skill.description == "Updated skill."

    async def test_create_version_nonexistent_skill_raises(self, skill_service):
        with pytest.raises(SkillNotFoundError):
            await skill_service.create_skill_version(
                CreateSkillVersionRequest(skill_id="nonexistent"),
                [_make_upload_file("SKILL.md", SAMPLE_MANIFEST)],
            )

    async def test_create_version_no_files_raises(self, skill_service):
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        with pytest.raises(ValueError, match="At least one file"):
            await skill_service.create_skill_version(CreateSkillVersionRequest(skill_id=created.id), [])

    async def test_multiple_versions_sequential(self, skill_service):
        """Creating multiple versions increments the version number correctly."""
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])

        for i in range(2, 5):
            ver = await skill_service.create_skill_version(
                CreateSkillVersionRequest(skill_id=created.id),
                [_make_upload_file("SKILL.md", SAMPLE_MANIFEST_V2)],
            )
            assert ver.version == i

        skill = await skill_service.get_skill(GetSkillRequest(skill_id=created.id))
        assert skill.latest_version == "4"


class TestListSkillVersions:
    """Tests for list_skill_versions method."""

    async def test_list_versions_returns_all(self, skill_service):
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        await skill_service.create_skill_version(
            CreateSkillVersionRequest(skill_id=created.id),
            [_make_upload_file("SKILL.md", SAMPLE_MANIFEST_V2)],
        )

        result = await skill_service.list_skill_versions(ListSkillVersionsRequest(skill_id=created.id))
        assert len(result.data) == 2

    async def test_list_versions_default_order_desc(self, skill_service):
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        await skill_service.create_skill_version(
            CreateSkillVersionRequest(skill_id=created.id),
            [_make_upload_file("SKILL.md", SAMPLE_MANIFEST_V2)],
        )

        result = await skill_service.list_skill_versions(ListSkillVersionsRequest(skill_id=created.id))
        assert result.data[0].version > result.data[-1].version

    async def test_list_versions_with_limit(self, skill_service):
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        for _ in range(3):
            await skill_service.create_skill_version(
                CreateSkillVersionRequest(skill_id=created.id),
                [_make_upload_file("SKILL.md", SAMPLE_MANIFEST_V2)],
            )

        result = await skill_service.list_skill_versions(ListSkillVersionsRequest(skill_id=created.id, limit=2))
        assert len(result.data) == 2
        assert result.has_more is True

    async def test_list_versions_nonexistent_skill_raises(self, skill_service):
        with pytest.raises(SkillNotFoundError):
            await skill_service.list_skill_versions(ListSkillVersionsRequest(skill_id="nonexistent"))


class TestGetSkillVersion:
    """Tests for get_skill_version method."""

    async def test_get_version_returns_version(self, skill_service):
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])

        ver = await skill_service.get_skill_version(GetSkillVersionRequest(skill_id=created.id, version=1))
        assert ver.version == 1
        assert ver.skill_id == created.id
        assert ver.name == "test-skill"

    async def test_get_version_not_found_raises(self, skill_service):
        created = await skill_service.create_skill([_make_upload_file("SKILL.md", SAMPLE_MANIFEST)])
        with pytest.raises(ValueError, match="not found"):
            await skill_service.get_skill_version(GetSkillVersionRequest(skill_id=created.id, version=99))

    async def test_get_version_nonexistent_skill_raises(self, skill_service):
        with pytest.raises(SkillNotFoundError):
            await skill_service.get_skill_version(GetSkillVersionRequest(skill_id="nonexistent", version=1))
