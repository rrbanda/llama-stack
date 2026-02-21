# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Skills API protocol definition.

This module contains the Skills protocol definition.
Pydantic models are defined in llama_stack_api.skills.models.
The FastAPI router is defined in llama_stack_api.skills.fastapi_routes.
"""

from typing import Protocol, runtime_checkable

from fastapi import Response, UploadFile

from .models import (
    DeletedSkill,
    DeleteSkillRequest,
    GetSkillContentRequest,
    GetSkillRequest,
    ListSkillsRequest,
    ListSkillsResponse,
    Skill,
    UpdateSkillRequest,
)


@runtime_checkable
class Skills(Protocol):
    """Protocol for skill bundle management operations."""

    async def create_skill(
        self,
        files: list[UploadFile],
    ) -> Skill: ...

    async def list_skills(
        self,
        request: ListSkillsRequest,
    ) -> ListSkillsResponse: ...

    async def get_skill(
        self,
        request: GetSkillRequest,
    ) -> Skill: ...

    async def update_skill(
        self,
        request: UpdateSkillRequest,
    ) -> Skill: ...

    async def delete_skill(
        self,
        request: DeleteSkillRequest,
    ) -> DeletedSkill: ...

    async def get_skill_content(
        self,
        request: GetSkillContentRequest,
    ) -> Response: ...
