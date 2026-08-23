"""Byte-stability tests: these guard the property every cached layer depends on."""

from hotset.corpus.models import Tool
from hotset.layout.serialize import canonical_tool, index_line, layer_a_index

CREATE_ISSUE = Tool(
    name="create_issue",
    description="Create a new issue in a GitHub repository.\n\nRequires write access.",
    input_schema={
        "type": "object",
        "properties": {"owner": {}, "repo": {}, "title": {}, "body": {}, "labels": {}},
        "required": ["owner", "repo", "title"],
    },
)


def test_index_line_shape():
    line = index_line(CREATE_ISSUE)
    assert line == (
        "create_issue(owner,repo,title,body?,labels?) - "
        "Create a new issue in a GitHub repository."
    )
    assert "\n" not in line


def test_index_line_handles_empty_description():
    assert index_line(Tool(name="noop")) == "noop() - "


def test_canonical_tool_is_key_order_invariant():
    """Logically identical schemas must produce identical bytes — the silent cache killer."""
    a = Tool(name="t", input_schema={"type": "object", "required": ["b", "a"]})
    b = Tool(name="t", input_schema={"required": ["a", "b"], "type": "object"})
    assert canonical_tool(a) == canonical_tool(b)


def test_layer_a_index_is_name_sorted():
    tools = [Tool(name="zed"), Tool(name="alpha"), Tool(name="mid")]
    assert [ln.split("(")[0] for ln in layer_a_index(tools).splitlines()] == ["alpha", "mid", "zed"]
