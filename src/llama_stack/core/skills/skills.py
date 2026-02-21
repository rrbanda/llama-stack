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
    DeletedSkill,
    DeleteSkillRequest,
    GetSkillContentRequest,
    GetSkillRequest,
    ListSkillsRequest,
    ListSkillsResponse,
    Order,
    Skill,
    SkillNotFoundError,
    Skills,
    UpdateSkillRequest,
)

logger = get_logger(name=__name__, category="skills")

KEY_PREFIX = "skills:v1:"


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
        """Get the KVStore key for a skill."""
        return f"{KEY_PREFIX}{skill_id}"

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
        """Get the filesystem directory for a skill's content."""
        return self.storage_dir / skill_id

    async def create_skill(self, files: list[UploadFile]) -> Skill:
        """Create a new skill from uploaded files.

        Accepts either a single zip archive or multiple files via multipart
        upload.  The bundle must contain exactly one SKILL.md manifest.
        Name and description are extracted from its YAML frontmatter.
        """
        if not files:
            raise ValueError("At least one file is required to create a skill")

        names: list[str] = []
        contents: list[bytes] = []
        for f in files:
            names.append(f.filename or "")
            contents.append(await f.read())

        # Single zip upload: extract the archive into a file map
        if len(files) == 1 and (
            (files[0].content_type and "zip" in files[0].content_type) or (names[0].lower().endswith(".zip"))
        ):
            file_map = extract_zip(contents[0])
        else:
            file_map = files_to_map(names, contents)

        validate_bundle(file_map)
        name, description = extract_metadata(file_map)

        skill_id = f"skill_{uuid.uuid4().hex[:24]}"
        created_at = int(time.time())

        skill_dir = self._skill_dir(skill_id)
        skill_dir.mkdir(parents=True, exist_ok=True)

        for relative_path, data in file_map.items():
            safe_parts = [p for p in PurePosixPath(relative_path).parts if p not in (".", "..")]
            dest = skill_dir / Path(*safe_parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)

        skill = Skill(
            id=skill_id,
            created_at=created_at,
            default_version="1",
            description=description,
            latest_version="1",
            name=name,
        )

        await self.kvstore.set(
            self._get_key(skill_id),
            json.dumps(skill.model_dump()),
        )

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
            return ListSkillsResponse(
                data=[],
                first_id="",
                has_more=False,
                last_id="",
            )

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
        skill.default_version = request.default_version

        await self.kvstore.set(
            self._get_key(request.skill_id),
            json.dumps(skill.model_dump()),
        )
        return skill

    async def delete_skill(self, request: DeleteSkillRequest) -> DeletedSkill:
        """Delete a skill and its stored content."""
        key = self._get_key(request.skill_id)
        if not await self.kvstore.get(key):
            raise SkillNotFoundError(request.skill_id)
        await self.kvstore.delete(key)

        skill_dir = self._skill_dir(request.skill_id)
        if skill_dir.exists():
            shutil.rmtree(skill_dir)

        return DeletedSkill(id=request.skill_id, deleted=True)

    async def get_skill_content(self, request: GetSkillContentRequest) -> Response:
        """Download a skill's content as a zip archive."""
        await self.get_skill(GetSkillRequest(skill_id=request.skill_id))

        skill_dir = self._skill_dir(request.skill_id)
        if not skill_dir.exists():
            raise SkillNotFoundError(request.skill_id)

        buffer = BytesIO()
        with ZipFile(buffer, "w") as zf:
            for file_path in sorted(skill_dir.rglob("*")):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(skill_dir))

        buffer.seek(0)
        return Response(
            content=buffer.read(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{request.skill_id}.zip"',
            },
        )

    async def shutdown(self):
        """Shutdown the skill service."""
        await self.kvstore.shutdown()
