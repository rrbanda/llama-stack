# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for the Skills API.

These tests exercise the Skills API through HTTP requests against a running
Llama Stack server. The llama_stack_client SDK does not yet have a ``skills``
namespace, so we use ``requests`` directly — the same pattern used by
``tests/integration/files/test_files.py``.
"""

import io
import zipfile

import pytest
import requests

MANIFEST_V1 = (
    b"---\nname: calculator\ndescription: A calculator skill.\n---\n# Calculator\nPerforms arithmetic operations.\n"
)

MANIFEST_V2 = (
    b"---\nname: calculator-v2\ndescription: An improved calculator.\n---\n"
    b"# Calculator v2\nPerforms advanced arithmetic operations.\n"
)

TIMEOUT = 30


@pytest.fixture()
def base_url(llama_stack_client, require_server):
    """Base URL for the Skills v1alpha API."""
    return f"{llama_stack_client.base_url}/v1alpha/skills"


@pytest.fixture()
def created_skill(base_url):
    """Create a skill and yield it, then clean up after the test."""
    resp = requests.post(
        base_url,
        files=[
            ("files", ("SKILL.md", MANIFEST_V1, "text/markdown")),
            ("files", ("run.sh", b"#!/bin/bash\necho hello\n", "text/x-shellscript")),
        ],
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    skill = resp.json()
    yield skill
    requests.delete(f"{base_url}/{skill['id']}", timeout=TIMEOUT)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def test_create_skill(base_url):
    """Creating a skill returns a valid Skill object with metadata from SKILL.md."""
    resp = requests.post(
        base_url,
        files=[
            ("files", ("SKILL.md", MANIFEST_V1, "text/markdown")),
            ("files", ("run.sh", b"#!/bin/bash\necho hello\n", "text/x-shellscript")),
        ],
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200, resp.text
    skill = resp.json()

    assert skill["id"].startswith("skill_")
    assert skill["name"] == "calculator"
    assert skill["description"] == "A calculator skill."
    assert skill["object"] == "skill"
    assert skill["latest_version"] == "1"
    assert skill["default_version"] == "1"
    assert "created_at" in skill

    requests.delete(f"{base_url}/{skill['id']}", timeout=TIMEOUT)


def test_create_skill_from_zip(base_url):
    """Creating a skill from a zip archive works correctly."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("SKILL.md", MANIFEST_V1.decode())
        zf.writestr("run.sh", "#!/bin/bash\necho hello\n")
        zf.writestr("lib/utils.sh", "# utility functions\n")
    buf.seek(0)

    resp = requests.post(
        base_url,
        files=[("files", ("skill-bundle.zip", buf.getvalue(), "application/zip"))],
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200, resp.text
    skill = resp.json()
    assert skill["name"] == "calculator"

    content_resp = requests.get(f"{base_url}/{skill['id']}/content", timeout=TIMEOUT)
    assert content_resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(content_resp.content)) as zf:
        names = zf.namelist()
        assert "SKILL.md" in names
        assert "run.sh" in names
        assert "lib/utils.sh" in names

    requests.delete(f"{base_url}/{skill['id']}", timeout=TIMEOUT)


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


def test_list_skills_empty(base_url):
    """Listing skills when none exist returns an empty list."""
    resp = requests.get(base_url, timeout=TIMEOUT)
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data


def test_list_skills_returns_created(base_url, created_skill):
    """A created skill appears in the list response."""
    resp = requests.get(base_url, timeout=TIMEOUT)
    assert resp.status_code == 200
    ids = [s["id"] for s in resp.json()["data"]]
    assert created_skill["id"] in ids


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------


def test_get_skill(base_url, created_skill):
    """Getting a skill by ID returns the correct skill."""
    resp = requests.get(f"{base_url}/{created_skill['id']}", timeout=TIMEOUT)
    assert resp.status_code == 200
    skill = resp.json()
    assert skill["id"] == created_skill["id"]
    assert skill["name"] == created_skill["name"]


def test_get_skill_not_found(base_url):
    """Getting a non-existent skill returns 404."""
    resp = requests.get(f"{base_url}/skill_nonexistent", timeout=TIMEOUT)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Content download
# ---------------------------------------------------------------------------


def test_get_skill_content(base_url, created_skill):
    """Downloading skill content returns a valid zip containing the uploaded files."""
    resp = requests.get(f"{base_url}/{created_skill['id']}/content", timeout=TIMEOUT)
    assert resp.status_code == 200
    assert "application/zip" in resp.headers.get("content-type", "")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        assert "SKILL.md" in names
        assert "run.sh" in names
        manifest = zf.read("SKILL.md")
        assert b"calculator" in manifest


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------


def test_create_version(base_url, created_skill):
    """Creating a new version increments the version number."""
    skill_id = created_skill["id"]

    resp = requests.post(
        f"{base_url}/{skill_id}/versions",
        files=[
            ("files", ("SKILL.md", MANIFEST_V2, "text/markdown")),
            ("files", ("run.sh", b"#!/bin/bash\necho improved\n", "text/x-shellscript")),
        ],
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200, resp.text
    version = resp.json()
    assert version["version"] == 2
    assert version["skill_id"] == skill_id
    assert version["name"] == "calculator-v2"
    assert version["object"] == "skill.version"

    skill_resp = requests.get(f"{base_url}/{skill_id}", timeout=TIMEOUT)
    updated_skill = skill_resp.json()
    assert updated_skill["latest_version"] == "2"


def test_list_versions(base_url, created_skill):
    """Listing versions returns all versions including the initial one."""
    skill_id = created_skill["id"]

    resp = requests.get(f"{base_url}/{skill_id}/versions", timeout=TIMEOUT)
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert len(data["data"]) >= 1
    assert data["data"][0]["version"] == 1


def test_get_specific_version(base_url, created_skill):
    """Getting a specific version returns the correct version metadata."""
    skill_id = created_skill["id"]

    resp = requests.get(f"{base_url}/{skill_id}/versions/1", timeout=TIMEOUT)
    assert resp.status_code == 200
    version = resp.json()
    assert version["version"] == 1
    assert version["skill_id"] == skill_id


def test_get_content_for_specific_version(base_url, created_skill):
    """Downloading content for a specific version returns that version's files."""
    skill_id = created_skill["id"]

    requests.post(
        f"{base_url}/{skill_id}/versions",
        files=[
            ("files", ("SKILL.md", MANIFEST_V2, "text/markdown")),
            ("files", ("run.sh", b"#!/bin/bash\necho v2\n", "text/x-shellscript")),
        ],
        timeout=TIMEOUT,
    )

    v1_resp = requests.get(f"{base_url}/{skill_id}/content?version=1", timeout=TIMEOUT)
    assert v1_resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(v1_resp.content)) as zf:
        manifest = zf.read("SKILL.md")
        assert b"calculator" in manifest

    v2_resp = requests.get(f"{base_url}/{skill_id}/content?version=2", timeout=TIMEOUT)
    assert v2_resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(v2_resp.content)) as zf:
        manifest = zf.read("SKILL.md")
        assert b"calculator-v2" in manifest


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


def test_update_default_version(base_url, created_skill):
    """Updating the default version changes the skill's default_version field."""
    skill_id = created_skill["id"]

    requests.post(
        f"{base_url}/{skill_id}/versions",
        files=[
            ("files", ("SKILL.md", MANIFEST_V2, "text/markdown")),
            ("files", ("run.sh", b"#!/bin/bash\necho v2\n", "text/x-shellscript")),
        ],
        timeout=TIMEOUT,
    )

    resp = requests.post(
        f"{base_url}/{skill_id}",
        json={"default_version": "2"},
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200, resp.text
    skill = resp.json()
    assert skill["default_version"] == "2"


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_skill(base_url):
    """Deleting a skill removes it and returns a DeletedSkill confirmation."""
    create_resp = requests.post(
        base_url,
        files=[
            ("files", ("SKILL.md", MANIFEST_V1, "text/markdown")),
            ("files", ("run.sh", b"echo hi", "text/x-shellscript")),
        ],
        timeout=TIMEOUT,
    )
    skill = create_resp.json()
    skill_id = skill["id"]

    del_resp = requests.delete(f"{base_url}/{skill_id}", timeout=TIMEOUT)
    assert del_resp.status_code == 200
    deleted = del_resp.json()
    assert deleted["id"] == skill_id
    assert deleted["deleted"] is True

    get_resp = requests.get(f"{base_url}/{skill_id}", timeout=TIMEOUT)
    assert get_resp.status_code == 404


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------


def test_full_lifecycle(base_url):
    """End-to-end lifecycle: create -> version -> update -> download -> delete."""
    create_resp = requests.post(
        base_url,
        files=[
            ("files", ("SKILL.md", MANIFEST_V1, "text/markdown")),
            ("files", ("run.sh", b"echo v1", "text/x-shellscript")),
        ],
        timeout=TIMEOUT,
    )
    assert create_resp.status_code == 200
    skill = create_resp.json()
    skill_id = skill["id"]

    try:
        assert skill["latest_version"] == "1"

        ver_resp = requests.post(
            f"{base_url}/{skill_id}/versions",
            files=[
                ("files", ("SKILL.md", MANIFEST_V2, "text/markdown")),
                ("files", ("run.sh", b"echo v2", "text/x-shellscript")),
            ],
            timeout=TIMEOUT,
        )
        assert ver_resp.status_code == 200
        assert ver_resp.json()["version"] == 2

        skill_resp = requests.get(f"{base_url}/{skill_id}", timeout=TIMEOUT)
        assert skill_resp.json()["latest_version"] == "2"
        assert skill_resp.json()["default_version"] == "1"

        upd_resp = requests.post(
            f"{base_url}/{skill_id}",
            json={"default_version": "2"},
            timeout=TIMEOUT,
        )
        assert upd_resp.status_code == 200
        assert upd_resp.json()["default_version"] == "2"

        content_resp = requests.get(
            f"{base_url}/{skill_id}/content",
            timeout=TIMEOUT,
        )
        assert content_resp.status_code == 200
        with zipfile.ZipFile(io.BytesIO(content_resp.content)) as zf:
            assert b"calculator-v2" in zf.read("SKILL.md")

        versions_resp = requests.get(
            f"{base_url}/{skill_id}/versions",
            timeout=TIMEOUT,
        )
        assert versions_resp.status_code == 200
        assert len(versions_resp.json()["data"]) == 2

    finally:
        requests.delete(f"{base_url}/{skill_id}", timeout=TIMEOUT)

    get_resp = requests.get(f"{base_url}/{skill_id}", timeout=TIMEOUT)
    assert get_resp.status_code == 404
