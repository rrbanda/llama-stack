# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


from llama_stack_api import (
    DeletedSkill,
    ListSkillsResponse,
    Skill,
    SkillNotFoundError,
    Skills,
)
from llama_stack_api.skills.fastapi_routes import create_router
from llama_stack_api.skills.models import (
    DeleteSkillRequest,
    GetSkillRequest,
    ListSkillsRequest,
    UpdateSkillRequest,
)


def _create_mock_skill(skill_id: str = "skill_abc123") -> Skill:
    """Create a mock skill for testing."""
    return Skill(
        id=skill_id,
        created_at=1700000000,
        default_version="1",
        description="A test skill",
        latest_version="1",
        name="test-skill",
    )


def _get_endpoint(router, path: str, method: str = "GET"):
    """Get an endpoint function from router by path and method."""
    return next(
        r.endpoint for r in router.routes if getattr(r, "path", None) == path and method in getattr(r, "methods", set())
    )


# --- List Skills Tests ---


async def test_list_skills_returns_empty_list():
    """Test listing skills when none are registered."""
    impl = AsyncMock(spec=Skills)
    impl.list_skills.return_value = ListSkillsResponse(data=[], first_id="", has_more=False, last_id="")

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    list_endpoint = _get_endpoint(router, "/v1alpha/skills", "GET")
    request = ListSkillsRequest()
    response = await list_endpoint(request=request)

    assert response.data == []
    assert response.has_more is False
    impl.list_skills.assert_awaited_once()


async def test_list_skills_returns_skills():
    """Test listing skills returns registered skills."""
    impl = AsyncMock(spec=Skills)
    mock_skill = _create_mock_skill()
    impl.list_skills.return_value = ListSkillsResponse(
        data=[mock_skill],
        first_id=mock_skill.id,
        has_more=False,
        last_id=mock_skill.id,
    )

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    list_endpoint = _get_endpoint(router, "/v1alpha/skills", "GET")
    request = ListSkillsRequest()
    response = await list_endpoint(request=request)

    assert len(response.data) == 1
    assert response.data[0].id == "skill_abc123"
    impl.list_skills.assert_awaited_once()


# --- Get Skill Tests ---


async def test_get_skill_returns_skill():
    """Test getting a skill by ID."""
    impl = AsyncMock(spec=Skills)
    mock_skill = _create_mock_skill()
    impl.get_skill.return_value = mock_skill

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    get_endpoint = _get_endpoint(router, "/v1alpha/skills/{skill_id}", "GET")
    request = GetSkillRequest(skill_id="skill_abc123")
    response = await get_endpoint(request=request)

    assert response.id == "skill_abc123"
    assert response.name == "test-skill"
    impl.get_skill.assert_awaited_once()


async def test_get_skill_not_found_raises_error():
    """Test getting a non-existent skill raises SkillNotFoundError."""
    impl = AsyncMock(spec=Skills)
    impl.get_skill.side_effect = SkillNotFoundError("nonexistent")

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    get_endpoint = _get_endpoint(router, "/v1alpha/skills/{skill_id}", "GET")
    request = GetSkillRequest(skill_id="nonexistent")

    with pytest.raises(SkillNotFoundError):
        await get_endpoint(request=request)


# --- Delete Skill Tests ---


async def test_delete_skill_returns_deleted():
    """Test deleting a skill returns DeletedSkill."""
    impl = AsyncMock(spec=Skills)
    impl.delete_skill.return_value = DeletedSkill(id="skill_abc123", deleted=True)

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    delete_endpoint = _get_endpoint(router, "/v1alpha/skills/{skill_id}", "DELETE")
    request = DeleteSkillRequest(skill_id="skill_abc123")
    response = await delete_endpoint(request=request)

    assert response.id == "skill_abc123"
    assert response.deleted is True
    impl.delete_skill.assert_awaited_once()


async def test_delete_skill_not_found_raises_error():
    """Test deleting a non-existent skill raises SkillNotFoundError."""
    impl = AsyncMock(spec=Skills)
    impl.delete_skill.side_effect = SkillNotFoundError("nonexistent")

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    delete_endpoint = _get_endpoint(router, "/v1alpha/skills/{skill_id}", "DELETE")
    request = DeleteSkillRequest(skill_id="nonexistent")

    with pytest.raises(SkillNotFoundError):
        await delete_endpoint(request=request)


# --- Update Skill Tests ---


async def test_update_skill_returns_updated():
    """Test updating a skill returns the updated Skill."""
    impl = AsyncMock(spec=Skills)
    updated_skill = _create_mock_skill()
    updated_skill.default_version = "2"
    impl.update_skill.return_value = updated_skill

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    update_endpoint = _get_endpoint(router, "/v1alpha/skills/{skill_id}", "POST")
    response = await update_endpoint(skill_id="skill_abc123", default_version="2")

    assert response.default_version == "2"
    impl.update_skill.assert_awaited_once()
    call_args = impl.update_skill.call_args
    request_arg = call_args[0][0]
    assert isinstance(request_arg, UpdateSkillRequest)
    assert request_arg.skill_id == "skill_abc123"
    assert request_arg.default_version == "2"


# --- OpenAPI Schema Tests ---


def test_openapi_schema_has_skills_endpoints():
    """Test that OpenAPI schema includes all skills endpoints."""
    impl = AsyncMock(spec=Skills)
    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    schema = app.openapi()

    assert "/v1alpha/skills" in schema["paths"]
    assert "/v1alpha/skills/{skill_id}" in schema["paths"]
    assert "/v1alpha/skills/{skill_id}/content" in schema["paths"]


def test_openapi_schema_list_skills_is_get():
    """Test list skills endpoint is documented as GET."""
    impl = AsyncMock(spec=Skills)
    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    schema = app.openapi()
    skills_path = schema["paths"]["/v1alpha/skills"]

    assert "get" in skills_path
    assert skills_path["get"]["summary"] == "List skills."


def test_openapi_schema_create_skill_is_post():
    """Test create skill endpoint is documented as POST."""
    impl = AsyncMock(spec=Skills)
    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    schema = app.openapi()
    skills_path = schema["paths"]["/v1alpha/skills"]

    assert "post" in skills_path
    assert skills_path["post"]["summary"] == "Create a skill."


def test_openapi_schema_get_skill_has_path_param():
    """Test get skill endpoint has skill_id path parameter."""
    impl = AsyncMock(spec=Skills)
    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    schema = app.openapi()
    get_skill_path = schema["paths"]["/v1alpha/skills/{skill_id}"]

    assert "get" in get_skill_path
    parameters = get_skill_path["get"]["parameters"]
    param_names = [p["name"] for p in parameters]
    assert "skill_id" in param_names
