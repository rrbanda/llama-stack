# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Skills API models.

This module contains the Pydantic models for the Skills API.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from llama_stack_api.common.responses import Order
from llama_stack_api.schema_utils import json_schema_type


@json_schema_type
class Skill(BaseModel):
    """A reusable skill bundle registered in Llama Stack."""

    id: str = Field(..., description="Unique identifier for the skill.")
    created_at: int = Field(..., description="Unix timestamp (seconds) for when the skill was created.")
    default_version: str = Field(..., description="Default version for the skill.")
    description: str = Field(..., description="Description of the skill.")
    latest_version: str = Field(..., description="Latest version for the skill.")
    name: str = Field(..., description="Name of the skill.")
    object: Literal["skill"] = Field(default="skill", description="The object type, which is always 'skill'.")


@json_schema_type
class DeletedSkill(BaseModel):
    """Response for a deleted skill."""

    id: str = Field(..., description="The skill identifier that was deleted.")
    deleted: bool = Field(..., description="Whether the skill was successfully deleted.")
    object: Literal["skill.deleted"] = Field(
        default="skill.deleted", description="The object type, which is always 'skill.deleted'."
    )


# Request models


@json_schema_type
class GetSkillRequest(BaseModel):
    """Request model for getting a skill by ID."""

    skill_id: str = Field(..., description="The ID of the skill to retrieve.")


@json_schema_type
class ListSkillsRequest(BaseModel):
    """Request model for listing skills."""

    order: Order | None = Field(default=Order.desc, description="Sort order by created_at timestamp ('asc' or 'desc').")
    limit: int | None = Field(default=None, description="Number of items to retrieve.")
    after: str | None = Field(
        default=None, description="Identifier for the last item from the previous pagination request."
    )


@json_schema_type
class UpdateSkillRequest(BaseModel):
    """Request model for updating a skill's default version."""

    skill_id: str = Field(..., description="The ID of the skill to update.")
    default_version: str = Field(..., description="The skill version number to set as default.")

    @field_validator("default_version")
    @classmethod
    def validate_default_version(cls, v: str) -> str:
        try:
            n = int(v)
        except (ValueError, TypeError):
            raise ValueError(f"default_version must be a numeric string, got {v!r}") from None
        if n < 1:
            raise ValueError(f"default_version must be a positive integer, got {v!r}")
        return v


@json_schema_type
class DeleteSkillRequest(BaseModel):
    """Request model for deleting a skill."""

    skill_id: str = Field(..., description="The ID of the skill to delete.")


@json_schema_type
class GetSkillContentRequest(BaseModel):
    """Request model for downloading a skill zip bundle."""

    skill_id: str = Field(..., description="The ID of the skill to download content for.")
    version: int | None = Field(
        default=None, description="Version to download. Defaults to the skill's default version."
    )


# --- Skill version models ---


@json_schema_type
class SkillVersion(BaseModel):
    """A specific version of a skill bundle."""

    version: int = Field(..., description="The version number.")
    created_at: int = Field(..., description="Unix timestamp (seconds) for when this version was created.")
    skill_id: str = Field(..., description="The ID of the parent skill.")
    name: str = Field(..., description="Name extracted from this version's SKILL.md frontmatter.")
    description: str = Field(..., description="Description extracted from this version's SKILL.md frontmatter.")
    object: Literal["skill.version"] = Field(
        default="skill.version", description="The object type, which is always 'skill.version'."
    )


@json_schema_type
class CreateSkillVersionRequest(BaseModel):
    """Request model for creating a new skill version."""

    skill_id: str = Field(..., description="The ID of the skill to create a version for.")


@json_schema_type
class ListSkillVersionsRequest(BaseModel):
    """Request model for listing versions of a skill."""

    skill_id: str = Field(..., description="The ID of the skill to list versions for.")
    order: Order | None = Field(default=Order.desc, description="Sort order by created_at timestamp ('asc' or 'desc').")
    limit: int | None = Field(default=None, description="Number of items to retrieve.")
    after: str | None = Field(
        default=None, description="Cursor for the last item from the previous pagination request."
    )


@json_schema_type
class GetSkillVersionRequest(BaseModel):
    """Request model for getting a specific skill version."""

    skill_id: str = Field(..., description="The ID of the skill.")
    version: int = Field(..., description="The version number to retrieve.")


# Response models


@json_schema_type
class ListSkillsResponse(BaseModel):
    """Paginated response containing a list of skills."""

    data: list[Skill] = Field(..., description="A list of skill items.")
    first_id: str = Field(..., description="The ID of the first item in the list.")
    has_more: bool = Field(..., description="Whether there are more items available.")
    last_id: str = Field(..., description="The ID of the last item in the list.")
    object: Literal["list"] = Field(default="list", description="The object type, which is always 'list'.")


@json_schema_type
class ListSkillVersionsResponse(BaseModel):
    """Paginated response containing a list of skill versions."""

    data: list[SkillVersion] = Field(..., description="A list of skill version items.")
    first_id: str = Field(..., description="The version number of the first item in the list.")
    has_more: bool = Field(..., description="Whether there are more items available.")
    last_id: str = Field(..., description="The version number of the last item in the list.")
    object: Literal["list"] = Field(default="list", description="The object type, which is always 'list'.")


__all__ = [
    "CreateSkillVersionRequest",
    "DeletedSkill",
    "DeleteSkillRequest",
    "GetSkillContentRequest",
    "GetSkillRequest",
    "GetSkillVersionRequest",
    "ListSkillsRequest",
    "ListSkillsResponse",
    "ListSkillVersionsRequest",
    "ListSkillVersionsResponse",
    "Skill",
    "SkillVersion",
    "UpdateSkillRequest",
]
