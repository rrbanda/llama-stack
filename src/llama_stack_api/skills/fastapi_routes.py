# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Skills API.

This module defines the FastAPI router for the Skills API using standard
FastAPI route decorators.
"""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path, UploadFile
from fastapi.param_functions import File
from fastapi.responses import Response

from llama_stack_api.router_utils import create_path_dependency, create_query_dependency, standard_responses
from llama_stack_api.version import LLAMA_STACK_API_V1ALPHA

from .api import Skills
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

# Dependency functions from Pydantic models
get_skill_request = create_path_dependency(GetSkillRequest)
get_delete_skill_request = create_path_dependency(DeleteSkillRequest)
get_skill_content_request = create_path_dependency(GetSkillContentRequest)
get_list_skills_request = create_query_dependency(ListSkillsRequest)


def create_router(impl: Skills) -> APIRouter:
    """Create a FastAPI router for the Skills API.

    Args:
        impl: The Skills implementation instance

    Returns:
        APIRouter configured for the Skills API
    """
    router = APIRouter(
        prefix=f"/{LLAMA_STACK_API_V1ALPHA}",
        tags=["Skills"],
        responses=standard_responses,
    )

    @router.post(
        "/skills",
        response_model=Skill,
        summary="Create a skill.",
        description="Upload files to create a new skill bundle.",
    )
    async def create_skill(
        files: Annotated[list[UploadFile], File(description="The files to upload for the skill bundle.")],
    ) -> Skill:
        return await impl.create_skill(files)

    @router.get(
        "/skills",
        response_model=ListSkillsResponse,
        summary="List skills.",
        description="List all registered skills.",
    )
    async def list_skills(
        request: Annotated[ListSkillsRequest, Depends(get_list_skills_request)],
    ) -> ListSkillsResponse:
        return await impl.list_skills(request)

    # Route order: more specific routes must come before less specific ones.
    # /skills/{skill_id}/content must come before /skills/{skill_id}

    @router.get(
        "/skills/{skill_id}/content",
        status_code=200,
        summary="Get skill content.",
        description="Download the skill bundle as a zip archive.",
        responses={
            200: {
                "description": "The skill bundle as a zip archive.",
                "content": {"application/zip": {}},
            },
        },
    )
    async def get_skill_content(
        request: Annotated[GetSkillContentRequest, Depends(get_skill_content_request)],
    ) -> Response:
        return await impl.get_skill_content(request)

    @router.get(
        "/skills/{skill_id}",
        response_model=Skill,
        summary="Get a skill.",
        description="Get a skill by its ID.",
    )
    async def get_skill(
        request: Annotated[GetSkillRequest, Depends(get_skill_request)],
    ) -> Skill:
        return await impl.get_skill(request)

    @router.post(
        "/skills/{skill_id}",
        response_model=Skill,
        summary="Update a skill.",
        description="Update a skill's default version.",
    )
    async def update_skill(
        skill_id: Annotated[str, Path(description="The ID of the skill to update.")],
        default_version: Annotated[str, Body(embed=True, description="The skill version number to set as default.")],
    ) -> Skill:
        request = UpdateSkillRequest(skill_id=skill_id, default_version=default_version)
        return await impl.update_skill(request)

    @router.delete(
        "/skills/{skill_id}",
        response_model=DeletedSkill,
        summary="Delete a skill.",
        description="Delete a skill by its ID.",
    )
    async def delete_skill(
        request: Annotated[DeleteSkillRequest, Depends(get_delete_skill_request)],
    ) -> DeletedSkill:
        return await impl.delete_skill(request)

    return router
