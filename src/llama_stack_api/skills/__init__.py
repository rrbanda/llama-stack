# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Skills API package.

This package contains the Skills API definition, models, and FastAPI router.
"""

from . import fastapi_routes
from .api import Skills
from .models import (
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
    Skill,
    SkillVersion,
    UpdateSkillRequest,
)

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
    "Skills",
    "UpdateSkillRequest",
    "fastapi_routes",
]
