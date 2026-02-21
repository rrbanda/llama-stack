# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os
import shutil
import time
import uuid
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from fastapi import Response, UploadFile
from pydantic import BaseModel, Field

from llama_stack.core.datatypes import StackConfig
from llama_stack.core.skills.bundle import (
    extract_metadata,
    extract_zip,
    files_to_map,
    validate_bundle,
)
from llama_stack.core.storage.kvstore import KVStore, kvstore_impl
from llama_stack.log import get_logger
from llama_stack_api import (
    CreateSkillVersionRequest,
    DeletedSkill,
    DeleteSkillRequest,
    GetSkillContentRequest,
    GetSkillRequest,
    GetSkillVersionRequest,
    ListSkillsRequest,
    ListSkillsResponse,
    ListSkillVersionsRequest,
    ListSkillVersionsResponse,
    Order,
    Skill,
    SkillNotFoundError,
    Skills,
    SkillVersion,
    UpdateSkillRequest,
)

logger = get_logger(name=__name__, category="skills")

KEY_PREFIX = "skills:v1:"
VERSION_KEY_PREFIX = "skillver:v1:"


class SkillServiceConfig(BaseModel):
    """Configuration for the built-in skill service."""

    config: StackConfig = Field(..., description="Stack run configuration for resolving persistence")


async def get_provider_impl(config: SkillServiceConfig):
    """Get the skill service implementation."""
    impl = SkillServiceImpl(config)
    return impl


class SkillServiceImpl(Skills):
    """Built-in skill service implementation."""

    def __init__(self, config: SkillServiceConfig):
        self.config = config
        self.kvstore: KVStore
        self.storage_dir: Path

    def _get_key(self, skill_id: str) -> str:
        return f"{KEY_PREFIX}{skill_id}"

    def _version_key(self, skill_id: str, version: int) -> str:
        return f"{VERSION_KEY_PREFIX}{skill_id}:{version}"

    def _version_key_prefix(self, skill_id: str) -> str:
        return f"{VERSION_KEY_PREFIX}{skill_id}:"

    async def initialize(self):
        """Initialize the skill service."""
        skills_ref = self.config.config.storage.stores.skills
        if not skills_ref:
            raise ValueError("storage.stores.skills must be configured in config")
        self.kvstore = await kvstore_impl(skills_ref)

        base_dir = os.path.expanduser(os.environ.get("SKILLS_STORAGE_DIR", "~/.llama/skills"))
        self.storage_dir = Path(base_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _skill_dir(self, skill_id: str) -> Path:
        return self.storage_dir / skill_id

    def _version_dir(self, skill_id: str, version: int) -> Path:
        return self.storage_dir / skill_id / f"v{version}"

    async def _build_file_map(self, files: list[UploadFile]) -> dict[str, bytes]:
        """Read uploaded files and produce a validated file map."""
        names: list[str] = []
        contents: list[bytes] = []
        for f in files:
            names.append(f.filename or "")
            contents.append(await f.read())

        if len(files) == 1 and (
            (files[0].content_type and "zip" in files[0].content_type) or (names[0].lower().endswith(".zip"))
        ):
            file_map = extract_zip(contents[0])
        else:
            file_map = files_to_map(names, contents)

        validate_bundle(file_map)
        return file_map

    def _write_files(self, dest_dir: Path, file_map: dict[str, bytes]) -> None:
        """Write a file map to a directory, preserving relative paths."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        for relative_path, data in file_map.items():
            safe_parts = [p for p in PurePosixPath(relative_path).parts if p not in (".", "..")]
            dest = dest_dir / Path(*safe_parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)

    def _zip_directory(self, source_dir: Path) -> bytes:
        """Create a zip archive from a directory's contents."""
        buffer = BytesIO()
        with ZipFile(buffer, "w") as zf:
            for file_path in sorted(source_dir.rglob("*")):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(source_dir))
        buffer.seek(0)
        return buffer.read()

    # --- Skill CRUD ---

    async def create_skill(self, files: list[UploadFile]) -> Skill:
        """Create a new skill from uploaded files.

        Accepts either a single zip archive or multiple files via multipart
        upload.  The bundle must contain exactly one SKILL.md manifest.
        Name and description are extracted from its YAML frontmatter.
        Creates version 1 automatically.
        """
        if not files:
            raise ValueError("At least one file is required to create a skill")

        file_map = await self._build_file_map(files)
        name, description = extract_metadata(file_map)

        skill_id = f"skill_{uuid.uuid4().hex[:24]}"
        created_at = int(time.time())

        self._write_files(self._version_dir(skill_id, 1), file_map)

        skill = Skill(
            id=skill_id,
            created_at=created_at,
            default_version="1",
            description=description,
            latest_version="1",
            name=name,
        )
        await self.kvstore.set(self._get_key(skill_id), json.dumps(skill.model_dump()))

        version = SkillVersion(
            version=1,
            created_at=created_at,
            skill_id=skill_id,
            name=name,
            description=description,
        )
        await self.kvstore.set(self._version_key(skill_id, 1), json.dumps(version.model_dump()))

        return skill

    async def list_skills(self, request: ListSkillsRequest) -> ListSkillsResponse:
        """List all skills with optional pagination."""
        keys = await self.kvstore.keys_in_range(KEY_PREFIX, KEY_PREFIX + "\uffff")

        skills: list[Skill] = []
        for key in keys:
            skill_json = await self.kvstore.get(key)
            if skill_json:
                skills.append(Skill.model_validate_json(skill_json))

        reverse = request.order != Order.asc
        skills.sort(key=lambda s: s.created_at, reverse=reverse)

        if request.after:
            idx = next((i for i, s in enumerate(skills) if s.id == request.after), None)
            if idx is not None:
                skills = skills[idx + 1 :]

        has_more = False
        if request.limit is not None and len(skills) > request.limit:
            skills = skills[: request.limit]
            has_more = True

        if not skills:
            return ListSkillsResponse(data=[], first_id="", has_more=False, last_id="")

        return ListSkillsResponse(
            data=skills,
            first_id=skills[0].id,
            has_more=has_more,
            last_id=skills[-1].id,
        )

    async def get_skill(self, request: GetSkillRequest) -> Skill:
        """Get a skill by its ID."""
        skill_json = await self.kvstore.get(self._get_key(request.skill_id))
        if not skill_json:
            raise SkillNotFoundError(request.skill_id)
        return Skill.model_validate_json(skill_json)

    async def update_skill(self, request: UpdateSkillRequest) -> Skill:
        """Update a skill's default version."""
        skill = await self.get_skill(GetSkillRequest(skill_id=request.skill_id))

        version_json = await self.kvstore.get(self._version_key(request.skill_id, int(request.default_version)))
        if not version_json:
            raise ValueError(f"Version {request.default_version} does not exist for skill {request.skill_id}")

        skill.default_version = request.default_version

        await self.kvstore.set(self._get_key(request.skill_id), json.dumps(skill.model_dump()))
        return skill

    async def delete_skill(self, request: DeleteSkillRequest) -> DeletedSkill:
        """Delete a skill and all its versions."""
        key = self._get_key(request.skill_id)
        if not await self.kvstore.get(key):
            raise SkillNotFoundError(request.skill_id)
        await self.kvstore.delete(key)

        ver_prefix = self._version_key_prefix(request.skill_id)
        ver_keys = await self.kvstore.keys_in_range(ver_prefix, ver_prefix + "\uffff")
        for vk in ver_keys:
            await self.kvstore.delete(vk)

        skill_dir = self._skill_dir(request.skill_id)
        if skill_dir.exists():
            shutil.rmtree(skill_dir)

        return DeletedSkill(id=request.skill_id, deleted=True)

    async def get_skill_content(self, request: GetSkillContentRequest) -> Response:
        """Download a skill version's content as a zip archive."""
        skill = await self.get_skill(GetSkillRequest(skill_id=request.skill_id))

        version = request.version if request.version is not None else int(skill.default_version)

        content_dir = self._version_dir(request.skill_id, version)
        if not content_dir.exists():
            raise SkillNotFoundError(request.skill_id)

        zip_bytes = self._zip_directory(content_dir)
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{request.skill_id}_v{version}.zip"',
            },
        )

    # --- Version management ---

    async def create_skill_version(self, request: CreateSkillVersionRequest, files: list[UploadFile]) -> SkillVersion:
        """Create a new version of an existing skill."""
        if not files:
            raise ValueError("At least one file is required to create a skill version")

        skill = await self.get_skill(GetSkillRequest(skill_id=request.skill_id))

        file_map = await self._build_file_map(files)
        name, description = extract_metadata(file_map)

        new_version = int(skill.latest_version) + 1
        created_at = int(time.time())

        self._write_files(self._version_dir(request.skill_id, new_version), file_map)

        version = SkillVersion(
            version=new_version,
            created_at=created_at,
            skill_id=request.skill_id,
            name=name,
            description=description,
        )
        await self.kvstore.set(
            self._version_key(request.skill_id, new_version),
            json.dumps(version.model_dump()),
        )

        skill.latest_version = str(new_version)
        skill.name = name
        skill.description = description
        await self.kvstore.set(self._get_key(request.skill_id), json.dumps(skill.model_dump()))

        return version

    async def list_skill_versions(self, request: ListSkillVersionsRequest) -> ListSkillVersionsResponse:
        """List all versions of a skill."""
        await self.get_skill(GetSkillRequest(skill_id=request.skill_id))

        prefix = self._version_key_prefix(request.skill_id)
        keys = await self.kvstore.keys_in_range(prefix, prefix + "\uffff")

        versions: list[SkillVersion] = []
        for key in keys:
            ver_json = await self.kvstore.get(key)
            if ver_json:
                versions.append(SkillVersion.model_validate_json(ver_json))

        reverse = request.order != Order.asc
        versions.sort(key=lambda v: v.version, reverse=reverse)

        if request.after:
            idx = next(
                (i for i, v in enumerate(versions) if str(v.version) == request.after),
                None,
            )
            if idx is not None:
                versions = versions[idx + 1 :]

        has_more = False
        if request.limit is not None and len(versions) > request.limit:
            versions = versions[: request.limit]
            has_more = True

        if not versions:
            return ListSkillVersionsResponse(data=[], first_id="", has_more=False, last_id="")

        return ListSkillVersionsResponse(
            data=versions,
            first_id=str(versions[0].version),
            has_more=has_more,
            last_id=str(versions[-1].version),
        )

    async def get_skill_version(self, request: GetSkillVersionRequest) -> SkillVersion:
        """Get a specific version of a skill."""
        await self.get_skill(GetSkillRequest(skill_id=request.skill_id))

        ver_json = await self.kvstore.get(self._version_key(request.skill_id, request.version))
        if not ver_json:
            raise ValueError(f"Version {request.version} not found for skill {request.skill_id}")
        return SkillVersion.model_validate_json(ver_json)

    async def shutdown(self):
        """Shutdown the skill service."""
        await self.kvstore.shutdown()
