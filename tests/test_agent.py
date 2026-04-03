"""Smoke tests for the τ²-Bench purple agent."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_extract_json_with_tags():
    from agent import extract_json_response
    text = '<json>{"name": "respond", "kwargs": {"content": "hello"}}</json>'
    result = extract_json_response(text)
    assert "<json>" in result
    parsed = json.loads(result.replace("<json>", "").replace("</json>", ""))
    assert parsed["name"] == "respond"


def test_extract_json_bare():
    from agent import extract_json_response
    text = 'I will call: {"name": "get_user", "kwargs": {"user_id": "123"}}'
    result = extract_json_response(text)
    parsed = json.loads(result.replace("<json>", "").replace("</json>", ""))
    assert parsed["name"] == "get_user"


def test_extract_json_fallback():
    from agent import extract_json_response
    text = "I'm sorry, I cannot help with that."
    result = extract_json_response(text)
    parsed = json.loads(result.replace("<json>", "").replace("</json>", ""))
    assert parsed["name"] == "respond"
    assert "cannot help" in parsed["kwargs"]["content"]
