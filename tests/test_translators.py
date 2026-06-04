"""Tests for aw.translators — both the agent-taking API and the *_from_spec cores.

These fill a coverage gap: before the extraction/rendering split the renderers
were exercised only by module doctests. They assert that each public function
equals its ``*_from_spec`` core fed the extracted spec (behavior preserved).
"""

from aw import LoadingAgent
from aw.translators import (
    claude_skill_from_spec,
    crewai_yaml_from_spec,
    extract_agent_spec,
    openai_assistant_from_spec,
    openai_tools_from_spec,
    to_claude_skill,
    to_crewai_yaml,
    to_openai_assistant,
    to_openai_tools,
)


def test_claude_skill_public_equals_from_spec():
    agent = LoadingAgent()
    spec = extract_agent_spec(agent, name="data-loading")
    assert to_claude_skill(agent, name="data-loading") == claude_skill_from_spec(
        spec, name="data-loading"
    )


def test_crewai_public_equals_from_spec():
    agent = LoadingAgent()
    spec = extract_agent_spec(agent, name="data_loader")
    assert to_crewai_yaml(agent, name="data_loader") == crewai_yaml_from_spec(
        spec, name="data_loader"
    )


def test_openai_tools_public_equals_from_spec():
    agent = LoadingAgent()
    spec = extract_agent_spec(agent)
    assert to_openai_tools(agent) == openai_tools_from_spec(spec)


def test_openai_assistant_public_equals_from_spec():
    agent = LoadingAgent()
    spec = extract_agent_spec(agent)
    assert to_openai_assistant(agent) == openai_assistant_from_spec(spec)


def test_from_spec_works_on_a_synthetic_spec():
    # The whole point of the split: drive renderers from a spec built elsewhere
    # (e.g. by coact) without a live aw agent.
    from aw.translators import AgentSpec, ToolSpec

    spec = AgentSpec(
        name="ext-agent",
        description="An externally built spec.",
        instructions="Do the thing.",
        tools=[ToolSpec(name="FileSamplerTool", description="Sample files.")],
        model="sonnet",
    )
    skill_md = claude_skill_from_spec(spec, name="ext-agent")
    assert "name: ext-agent" in skill_md
    crew = crewai_yaml_from_spec(spec, name="ext_agent")
    assert "ext_agent" in crew
    assert isinstance(openai_tools_from_spec(spec), list)
    assert "instructions" in openai_assistant_from_spec(spec)
