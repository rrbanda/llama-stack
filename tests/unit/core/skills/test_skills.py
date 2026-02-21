# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the Skills API implementation."""

import tempfile
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock
from zipfile import ZipFile

import pytest

from llama_stack.core.skills.skills import (
    KEY_PREFIX,
    SkillServiceImpl,
)
from llama_stack_api import (
    DeleteSkillRequest,
    GetSkillContentRequest,
    GetSkillRequest,
    ListSkillsRequest,
    Order,
    Skill,
    SkillNotFoundError,
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
    from pathlib import Path

    mock_config = MagicMock()
    service = SkillServiceImpl(mock_config)
    service.kvstore = mock_kvstore
    service.storage_dir = Path(storage_dir)
    return service


def _make_upload_file(filename: str, content: bytes) -> MagicMock:
    """Create a mock UploadFile."""
    upload_file = AsyncMock()
    upload_file.filename = filename
    upload_file.read = AsyncMock(return_value=content)
    return upload_file


# --- create_skill tests ---


class TestCreateSkill:
    """Tests for create_skill method."""

    async def test_create_skill_returns_skill(self, skill_service):
        """Test creating a skill returns a Skill object."""
        files = [_make_upload_file("SKILL.md", b"# My Skill")]
        result = await skill_service.create_skill(files)

        assert isinstance(result, Skill)
        assert result.id.startswith("skill_")
        assert result.name == "SKILL"
        assert result.default_version == "1"
        assert result.latest_version == "1"
        assert result.object == "skill"

    async def test_create_skill_stores_in_kvstore(self, skill_service, mock_kvstore):
        """Test creating a skill persists metadata to KVStore."""
        files = [_make_upload_file("SKILL.md", b"# My Skill")]
        result = await skill_service.create_skill(files)

        stored = await mock_kvstore.get(f"{KEY_PREFIX}{result.id}")
        assert stored is not None

    async def test_create_skill_stores_files_on_disk(self, skill_service, storage_dir):
        """Test creating a skill writes files to the storage directory."""
        from pathlib import Path

        files = [
            _make_upload_file("SKILL.md", b"# My Skill"),
            _make_upload_file("run.sh", b"#!/bin/bash\necho hello"),
        ]
        result = await skill_service.create_skill(files)

        skill_dir = Path(storage_dir) / result.id
        assert skill_dir.exists()
        assert (skill_dir / "SKILL.md").exists()
        assert (skill_dir / "run.sh").exists()
        assert (skill_dir / "SKILL.md").read_bytes() == b"# My Skill"

    async def test_create_skill_derives_name_from_first_file(self, skill_service):
        """Test skill name is derived from the first uploaded file."""
        files = [_make_upload_file("my_cool_skill.md", b"content")]
        result = await skill_service.create_skill(files)
        assert result.name == "my_cool_skill"

    async def test_create_skill_no_files_raises(self, skill_service):
        """Test creating a skill with no files raises ValueError."""
        with pytest.raises(ValueError, match="At least one file"):
            await skill_service.create_skill([])

    async def test_create_skill_sanitizes_path_traversal(self, skill_service, storage_dir):
        """Test that directory traversal in filenames is stripped."""
        from pathlib import Path

        files = [_make_upload_file("../../etc/passwd", b"malicious")]
        result = await skill_service.create_skill(files)

        skill_dir = Path(storage_dir) / result.id
        assert (skill_dir / "passwd").exists()
        assert not (Path(storage_dir) / ".." / ".." / "etc" / "passwd").exists()


# --- list_skills tests ---


class TestListSkills:
    """Tests for list_skills method."""

    async def test_list_skills_empty(self, skill_service):
        """Test listing skills when none exist."""
        result = await skill_service.list_skills(ListSkillsRequest())
        assert result.data == []
        assert result.has_more is False

    async def test_list_skills_returns_all(self, skill_service):
        """Test listing returns all created skills."""
        files = [_make_upload_file("SKILL.md", b"content")]
        await skill_service.create_skill(files)
        files = [_make_upload_file("SKILL.md", b"content")]
        await skill_service.create_skill(files)
        files = [_make_upload_file("SKILL.md", b"content")]
        await skill_service.create_skill(files)

        result = await skill_service.list_skills(ListSkillsRequest())
        assert len(result.data) == 3

    async def test_list_skills_default_order_is_desc(self, skill_service):
        """Test listing skills defaults to descending order by created_at."""
        files = [_make_upload_file("first.md", b"1")]
        await skill_service.create_skill(files)
        files = [_make_upload_file("second.md", b"2")]
        await skill_service.create_skill(files)

        result = await skill_service.list_skills(ListSkillsRequest())
        assert result.data[0].created_at >= result.data[-1].created_at

    async def test_list_skills_with_limit(self, skill_service):
        """Test listing skills with a limit."""
        for i in range(5):
            files = [_make_upload_file(f"skill_{i}.md", b"content")]
            await skill_service.create_skill(files)

        result = await skill_service.list_skills(ListSkillsRequest(limit=2))
        assert len(result.data) == 2
        assert result.has_more is True

    async def test_list_skills_with_after(self, skill_service):
        """Test listing skills with cursor-based pagination."""
        created = []
        for i in range(3):
            files = [_make_upload_file(f"skill_{i}.md", b"content")]
            created.append(await skill_service.create_skill(files))

        first_page = await skill_service.list_skills(ListSkillsRequest(limit=1))
        second_page = await skill_service.list_skills(ListSkillsRequest(after=first_page.last_id))
        assert second_page.data[0].id != first_page.data[0].id

    async def test_list_skills_asc_order(self, skill_service):
        """Test listing skills in ascending order."""
        files = [_make_upload_file("first.md", b"1")]
        await skill_service.create_skill(files)
        files = [_make_upload_file("second.md", b"2")]
        await skill_service.create_skill(files)

        result = await skill_service.list_skills(ListSkillsRequest(order=Order.asc))
        assert result.data[0].created_at <= result.data[-1].created_at


# --- get_skill tests ---


class TestGetSkill:
    """Tests for get_skill method."""

    async def test_get_skill_returns_skill(self, skill_service):
        """Test getting a skill by ID."""
        files = [_make_upload_file("SKILL.md", b"content")]
        created = await skill_service.create_skill(files)

        result = await skill_service.get_skill(GetSkillRequest(skill_id=created.id))
        assert result.id == created.id
        assert result.name == created.name

    async def test_get_skill_not_found(self, skill_service):
        """Test getting a non-existent skill raises error."""
        with pytest.raises(SkillNotFoundError):
            await skill_service.get_skill(GetSkillRequest(skill_id="nonexistent"))


# --- update_skill tests ---


class TestUpdateSkill:
    """Tests for update_skill method."""

    async def test_update_default_version(self, skill_service):
        """Test updating a skill's default version."""
        files = [_make_upload_file("SKILL.md", b"content")]
        created = await skill_service.create_skill(files)

        result = await skill_service.update_skill(UpdateSkillRequest(skill_id=created.id, default_version="2"))
        assert result.default_version == "2"

    async def test_update_persists_change(self, skill_service):
        """Test that updated default_version is persisted."""
        files = [_make_upload_file("SKILL.md", b"content")]
        created = await skill_service.create_skill(files)

        await skill_service.update_skill(UpdateSkillRequest(skill_id=created.id, default_version="3"))

        fetched = await skill_service.get_skill(GetSkillRequest(skill_id=created.id))
        assert fetched.default_version == "3"

    async def test_update_nonexistent_skill_raises(self, skill_service):
        """Test updating a non-existent skill raises error."""
        with pytest.raises(SkillNotFoundError):
            await skill_service.update_skill(UpdateSkillRequest(skill_id="nonexistent", default_version="1"))


# --- delete_skill tests ---


class TestDeleteSkill:
    """Tests for delete_skill method."""

    async def test_delete_skill_returns_deleted(self, skill_service):
        """Test deleting a skill returns a DeletedSkill object."""
        files = [_make_upload_file("SKILL.md", b"content")]
        created = await skill_service.create_skill(files)

        result = await skill_service.delete_skill(DeleteSkillRequest(skill_id=created.id))
        assert result.id == created.id
        assert result.deleted is True
        assert result.object == "skill.deleted"

    async def test_delete_skill_removes_from_kvstore(self, skill_service, mock_kvstore):
        """Test deleting a skill removes it from the store."""
        files = [_make_upload_file("SKILL.md", b"content")]
        created = await skill_service.create_skill(files)

        await skill_service.delete_skill(DeleteSkillRequest(skill_id=created.id))
        assert await mock_kvstore.get(f"{KEY_PREFIX}{created.id}") is None

    async def test_delete_skill_removes_files(self, skill_service, storage_dir):
        """Test deleting a skill removes its files from disk."""
        from pathlib import Path

        files = [_make_upload_file("SKILL.md", b"content")]
        created = await skill_service.create_skill(files)

        skill_dir = Path(storage_dir) / created.id
        assert skill_dir.exists()

        await skill_service.delete_skill(DeleteSkillRequest(skill_id=created.id))
        assert not skill_dir.exists()

    async def test_delete_nonexistent_skill_raises(self, skill_service):
        """Test deleting a non-existent skill raises error."""
        with pytest.raises(SkillNotFoundError):
            await skill_service.delete_skill(DeleteSkillRequest(skill_id="nonexistent"))

    async def test_delete_skill_not_in_list(self, skill_service):
        """Test deleted skill does not appear in list."""
        files = [_make_upload_file("SKILL.md", b"content")]
        created = await skill_service.create_skill(files)

        await skill_service.delete_skill(DeleteSkillRequest(skill_id=created.id))

        result = await skill_service.list_skills(ListSkillsRequest())
        assert len(result.data) == 0


# --- get_skill_content tests ---


class TestGetSkillContent:
    """Tests for get_skill_content method."""

    async def test_get_content_returns_zip(self, skill_service):
        """Test getting skill content returns a zip archive."""
        files = [
            _make_upload_file("SKILL.md", b"# My Skill"),
            _make_upload_file("run.sh", b"#!/bin/bash\necho hello"),
        ]
        created = await skill_service.create_skill(files)

        response = await skill_service.get_skill_content(GetSkillContentRequest(skill_id=created.id))

        assert response.media_type == "application/zip"
        zf = ZipFile(BytesIO(response.body))
        assert "SKILL.md" in zf.namelist()
        assert "run.sh" in zf.namelist()
        assert zf.read("SKILL.md") == b"# My Skill"

    async def test_get_content_nonexistent_skill_raises(self, skill_service):
        """Test getting content for a non-existent skill raises error."""
        with pytest.raises(SkillNotFoundError):
            await skill_service.get_skill_content(GetSkillContentRequest(skill_id="nonexistent"))


# --- Key prefix tests ---


class TestKeyPrefix:
    """Tests for skill key namespacing."""

    async def test_skills_use_namespaced_keys(self, skill_service, mock_kvstore):
        """Test that skills are stored with the correct key prefix."""
        files = [_make_upload_file("SKILL.md", b"content")]
        result = await skill_service.create_skill(files)

        keys = list(mock_kvstore._storage.keys())
        assert len(keys) == 1
        assert keys[0] == f"skills:v1:{result.id}"
