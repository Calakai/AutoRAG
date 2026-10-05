"""Static guards for the Claude Code plugin (no Claude needed)."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugin"
TOOL_PREFIX = "mcp__plugin_autorag_autorag__"
READ_TOOLS = {"search_knowledge", "read_chunk", "list_collections", "list_documents"}


def frontmatter(path: Path) -> dict[str, str]:
    match = re.match(r"^---\n(.*?)\n---\n", path.read_text(), re.S)
    assert match, f"{path} has no frontmatter"
    fields = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def test_manifests_agree():
    plugin = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    entry = next(p for p in market["plugins"] if p["name"] == plugin["name"])
    assert (ROOT / entry["source"]).resolve() == PLUGIN
    server = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"]["autorag"]
    assert [server["command"], *server["args"]] == ["autorag", "mcp"]


def test_mcp_tool_names_match_the_server():
    pytest.importorskip("mcp")
    from autorag import mcp_server

    source = (ROOT / "src" / "autorag" / "mcp_server.py").read_text()
    served = set(re.findall(r'@server\.tool\(name="(\w+)"', source))
    referenced = set()
    for md in PLUGIN.rglob("*.md"):
        referenced |= set(re.findall(TOOL_PREFIX + r"(\w+)", md.read_text()))
    assert referenced and referenced <= served, referenced - served
    assert mcp_server  # imported to prove the module loads


def test_librarian_is_read_only():
    tools = {t.strip() for t in frontmatter(PLUGIN / "agents" / "librarian.md")["tools"].split(",")}
    assert tools == {TOOL_PREFIX + t for t in READ_TOOLS}


def test_skills_reference_namespaced_agents_and_ingest_is_user_only():
    # A bare `agent: librarian` silently falls back to an unrestricted fork:
    # plugin agents are only found by their namespaced name.
    ask = frontmatter(PLUGIN / "skills" / "ask-docs" / "SKILL.md")
    assert ask["context"] == "fork" and ask["agent"] == "autorag:librarian"
    ingest = frontmatter(PLUGIN / "skills" / "ingest" / "SKILL.md")
    assert ingest["disable-model-invocation"] == "true"
    for md in PLUGIN.rglob("*.md"):
        fm = frontmatter(md)
        assert len(fm["description"]) <= 800, md
