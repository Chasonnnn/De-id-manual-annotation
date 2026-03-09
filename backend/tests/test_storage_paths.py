import json
from pathlib import Path

import server
from fastapi.testclient import TestClient

from models import CanonicalDocument, UtteranceRow


def test_runtime_storage_is_anchored_to_backend_dir():
    backend_dir = Path(server.__file__).resolve().parent

    assert server.BASE_DIR.is_absolute()
    assert server.BASE_DIR == backend_dir / ".annotation_tool"
    assert server.SESSIONS_DIR == server.BASE_DIR / "sessions"
    assert server.CONFIG_PATH == server.BASE_DIR / "config.json"
    assert server.PROFILE_PATH == server.BASE_DIR / "session_profile.json"


def test_legacy_repo_root_storage_is_migrated_to_backend_storage(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    backend_dir = repo_root / "backend"
    backend_dir.mkdir(parents=True)

    legacy_base_dir = repo_root / ".annotation_tool"
    legacy_session_dir = legacy_base_dir / "sessions" / "default"
    legacy_session_dir.mkdir(parents=True)

    doc = CanonicalDocument(
        id="legacy-doc",
        filename="legacy.json",
        format="hips_v1",
        raw_text="Hello Anna",
        utterances=[
            UtteranceRow(
                speaker="unknown",
                text="Hello Anna",
                global_start=0,
                global_end=10,
            )
        ],
        pre_annotations=[],
        label_set=[],
    )
    (legacy_session_dir / "_index.json").write_text(json.dumps([doc.id]))
    (legacy_session_dir / f"{doc.id}.source.json").write_text(doc.model_dump_json(indent=2))

    backend_base_dir = backend_dir / ".annotation_tool"
    monkeypatch.setattr("server.BACKEND_DIR", backend_dir)
    monkeypatch.setattr("server.REPO_ROOT", repo_root)
    monkeypatch.setattr("server.ROOT_ENV_PATH", repo_root / ".env.local")
    monkeypatch.setattr("server.BASE_DIR", backend_base_dir)
    monkeypatch.setattr("server.LEGACY_BASE_DIR", legacy_base_dir)
    monkeypatch.setattr("server.SESSIONS_DIR", backend_base_dir / "sessions")
    monkeypatch.setattr("server.CONFIG_PATH", backend_base_dir / "config.json")
    monkeypatch.setattr("server.PROFILE_PATH", backend_base_dir / "session_profile.json")
    server._session_docs.clear()

    client = TestClient(server.app)
    resp = client.get("/api/documents")

    assert resp.status_code == 200
    assert resp.json() == [{"id": "legacy-doc", "filename": "legacy.json", "status": "pending"}]
    assert (backend_base_dir / "sessions" / "default" / "_index.json").exists()
