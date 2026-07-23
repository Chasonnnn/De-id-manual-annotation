import agent
from fastapi.testclient import TestClient

from server import app


OFFERED_METHOD_IDS = ["dual", "deid_pipeline_cascade_gemma31b"]
RETIRED_METHOD_IDS = {
    "default",
    "extended",
    "verified",
    "dual-split",
    "presidio",
    "presidio+default",
    "presidio+llm-split",
    "deid_pipeline_deberta",
    "deid_pipeline_modernbert",
    "deid_pipeline_union",
    "deid_pipeline_cascade_gemma12b",
}


def test_agent_methods_endpoint_offers_only_approved_methods(monkeypatch):
    unavailable_reason = "Synthetic Gemma 31B runtime unavailable"
    monkeypatch.setattr(
        agent,
        "_deid_pipeline_runtime_error",
        lambda method_id=None: (
            unavailable_reason
            if method_id == "deid_pipeline_cascade_gemma31b"
            else None
        ),
    )

    response = TestClient(app).get("/api/agent/methods")

    assert response.status_code == 200
    methods = response.json()["methods"]
    assert [method["id"] for method in methods] == OFFERED_METHOD_IDS
    assert RETIRED_METHOD_IDS.isdisjoint(method["id"] for method in methods)
    assert [method["available"] for method in methods] == [True, False]
    assert methods[0]["unavailable_reason"] is None
    assert methods[1]["unavailable_reason"] == unavailable_reason


def test_offered_methods_keep_catalog_and_definition_payloads(monkeypatch):
    monkeypatch.setattr(agent, "_deid_pipeline_runtime_error", lambda method_id=None: None)

    response = TestClient(app).get("/api/agent/methods")

    assert response.status_code == 200
    catalog = {method["id"]: method for method in response.json()["methods"]}
    assert catalog["dual"] == {
        "id": "dual",
        "label": "Dual",
        "description": "Two-pass LLM (names + remaining explicit identifiers).",
        "requires_presidio": False,
        "uses_llm": True,
        "supports_verify_override": True,
        "default_verify": False,
        "prompt_templates": catalog["dual"]["prompt_templates"],
        "output_labels": list(agent.SIMPLE_LABELS),
        "known_limitations": [],
        "available": True,
        "unavailable_reason": None,
    }
    assert len(catalog["dual"]["prompt_templates"]) == 2
    assert catalog["deid_pipeline_cascade_gemma31b"] == {
        "id": "deid_pipeline_cascade_gemma31b",
        "label": "de-id pipeline · Operational union + Gemma 31B reviewer",
        "description": (
            "Operational Phase 21 union filtered by the 3-epoch Gemma 4 31B "
            "Phase31 reviewer (mlx-lm words, v2_current base + words "
            "institution/location rule + direct-address guard)."
        ),
        "requires_presidio": False,
        "uses_llm": False,
        "supports_verify_override": False,
        "default_verify": False,
        "prompt_templates": [],
        "output_labels": list(agent.SIMPLE_LABELS),
        "known_limitations": catalog["deid_pipeline_cascade_gemma31b"][
            "known_limitations"
        ],
        "available": True,
        "unavailable_reason": None,
    }
    assert catalog["deid_pipeline_cascade_gemma31b"]["known_limitations"]

    audited_definitions = {
        method["id"]: method
        for method in agent.get_method_definitions(method_bundle="audited")
    }
    assert RETIRED_METHOD_IDS <= audited_definitions.keys()
    cascade_definition = audited_definitions["deid_pipeline_cascade_gemma31b"]
    assert cascade_definition["output_labels_by_profile"] == {
        "simple": list(agent.SIMPLE_LABELS),
        "advanced": list(agent.ADVANCED_LABELS),
    }
