"""Translators from aw agent specifications to other agent formats.

Supports:
- Claude Code Skills / Agent Skills Open Standard (SKILL.md + scripts/)
- CrewAI (YAML agent/task configs)
- OpenAI tool schemas (JSON)

The core pattern is:
1. extract_agent_spec() introspects an aw agent into a normalized dict
2. Format-specific renderers convert the spec to target formats

Example:
    >>> from aw import LoadingAgent
    >>> from aw.translators import to_claude_skill
    >>> agent = LoadingAgent()
    >>> skill_md = to_claude_skill(agent, name='data-loading')
    >>> print(skill_md)  # doctest: +SKIP

    >>> # Write a complete skill directory
    >>> from aw.translators import write_skill_directory
    >>> write_skill_directory(agent, '/path/to/skills/data-loading')  # doctest: +SKIP
"""

import inspect
import json
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Union


# ---------------------------------------------------------------------------
# Intermediate representation: AgentSpec
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """Normalized description of a tool."""

    name: str
    description: str = ""
    parameters: dict = field(default_factory=dict)

    # For tools that can be rendered as scripts
    source: str = ""


@dataclass
class ValidatorSpec:
    """Normalized description of a validator."""

    name: str
    description: str = ""
    checks: list = field(default_factory=list)


@dataclass
class AgentSpec:
    """Normalized, format-agnostic description of an aw agent.

    This is the intermediate representation that all translators consume.
    It captures the essential components of an agent without being tied
    to any specific framework.

    Example:
        >>> from aw import LoadingAgent
        >>> spec = extract_agent_spec(LoadingAgent())
        >>> spec.name
        'LoadingAgent'
    """

    name: str
    description: str = ""
    instructions: str = ""
    tools: list = field(default_factory=list)  # list[ToolSpec]
    validators: list = field(default_factory=list)  # list[ValidatorSpec]
    model: str = ""
    max_retries: int = 3
    human_in_loop: bool = False
    source_class: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extraction: aw agent -> AgentSpec
# ---------------------------------------------------------------------------


def _extract_tool_spec(tool: Any) -> ToolSpec:
    """Extract a ToolSpec from a callable tool object."""
    name = getattr(tool, "__class__", type(tool)).__name__
    if name in ("function", "method"):
        name = getattr(tool, "__name__", "unknown_tool")

    doc = ""
    if hasattr(tool, "__doc__") and tool.__doc__:
        doc = inspect.cleandoc(tool.__doc__)
    elif hasattr(tool, "__class__") and tool.__class__.__doc__:
        doc = inspect.cleandoc(tool.__class__.__doc__)

    # Extract first paragraph as description
    description = doc.split("\n\n")[0] if doc else ""

    # Try to extract parameter info
    parameters = {}
    if hasattr(tool, "__init__"):
        sig = inspect.signature(tool.__init__)
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            parameters[pname] = {
                "default": (
                    repr(param.default)
                    if param.default is not inspect.Parameter.empty
                    else None
                ),
            }

    # Try to get source code for script generation
    source = ""
    try:
        source = inspect.getsource(tool.__class__)
    except (TypeError, OSError):
        pass

    return ToolSpec(
        name=name, description=description, parameters=parameters, source=source
    )


def _extract_validator_spec(validator: Any) -> ValidatorSpec:
    """Extract a ValidatorSpec from a validator callable."""
    name = getattr(validator, "__name__", None)
    if name is None or name == "<lambda>":
        name = getattr(validator, "__class__", type(validator)).__name__

    doc = getattr(validator, "__doc__", "") or ""
    if doc:
        doc = inspect.cleandoc(doc)

    # For composite validators (all_validators, any_validator), try to
    # extract the component validators from the closure
    checks = []
    if hasattr(validator, "__closure__") and validator.__closure__:
        for cell in validator.__closure__:
            try:
                val = cell.cell_contents
                if isinstance(val, (list, tuple)):
                    for item in val:
                        if callable(item):
                            # Try to get a meaningful name. Inner closures
                            # from is_type/is_not_empty are all named
                            # 'validate', so look at the qualname instead.
                            qualname = getattr(item, "__qualname__", "") or ""
                            if ".<locals>." in qualname:
                                # e.g. 'is_type.<locals>.validate' -> 'is_type'
                                outer_name = qualname.split(".<locals>.")[0]
                                checks.append(outer_name)
                            else:
                                inner_name = getattr(item, "__name__", None)
                                if inner_name and inner_name != "<lambda>":
                                    checks.append(inner_name)
            except ValueError:
                pass

    # Deduplicate while preserving order
    seen = set()
    unique_checks = []
    for c in checks:
        if c not in seen:
            seen.add(c)
            unique_checks.append(c)

    return ValidatorSpec(name=name, description=doc, checks=unique_checks)


def _infer_instructions(agent: Any) -> str:
    """Infer natural language instructions from an agent's structure.

    Examines the agent's methods, docstrings, and code to produce
    a human-readable description of what the agent does.
    """
    lines = []

    # Extract class docstring as primary instruction source
    class_doc = inspect.getdoc(agent)
    if class_doc:
        lines.append(class_doc)
        lines.append("")

    # Look for execute method docstring
    if hasattr(agent, "execute"):
        execute_doc = inspect.getdoc(agent.execute)
        if execute_doc and execute_doc != class_doc:
            lines.append("### Execution")
            lines.append(execute_doc)
            lines.append("")

    # Look for code generation methods (common in aw agents)
    for method_name in dir(agent):
        if method_name.startswith("_generate_") and not method_name.startswith("__"):
            method = getattr(agent, method_name, None)
            if method and callable(method):
                doc = inspect.getdoc(method)
                if doc:
                    # Convert method name to readable form
                    readable = method_name.replace("_generate_", "").replace("_", " ")
                    lines.append(f"- **{readable}**: {doc.split(chr(10))[0]}")

    return "\n".join(lines)


def extract_agent_spec(agent: Any, name: str = None) -> AgentSpec:
    """Extract a normalized AgentSpec from any aw agent.

    Introspects the agent's class, config, tools, and validators
    to build a format-agnostic intermediate representation.

    Args:
        agent: An aw agent instance (LoadingAgent, PreparationAgent, etc.)
        name: Override name (defaults to class name)

    Returns:
        AgentSpec with all extractable information

    Example:
        >>> from aw import LoadingAgent
        >>> spec = extract_agent_spec(LoadingAgent())
        >>> spec.name
        'LoadingAgent'
        >>> len(spec.tools) >= 1
        True
    """
    # Name
    agent_name = name or agent.__class__.__name__

    # Description from docstring
    description = ""
    if agent.__class__.__doc__:
        doc = inspect.cleandoc(agent.__class__.__doc__)
        # First paragraph as description
        description = doc.split("\n\n")[0]

    # Instructions from deeper analysis
    instructions = _infer_instructions(agent)

    # Config
    config = getattr(agent, "config", None)
    model = ""
    max_retries = 3
    human_in_loop = False
    if config:
        llm = getattr(config, "llm", "")
        model = llm if isinstance(llm, str) else ""
        max_retries = getattr(config, "max_retries", 3)
        human_in_loop = getattr(config, "human_in_loop", False)

    # Tools: look for tool attributes on the agent
    tools = []
    tool_attr_names = []
    for attr_name in dir(agent):
        if attr_name.startswith("_"):
            continue
        attr = getattr(agent, attr_name, None)
        if attr is None or isinstance(attr, (str, int, float, bool)):
            continue
        # Heuristic: it's a tool if it's callable and its class name ends with
        # 'Tool' or it's in the config.tools list
        cls_name = getattr(type(attr), "__name__", "")
        if cls_name.endswith("Tool"):
            tools.append(_extract_tool_spec(attr))
            tool_attr_names.append(attr_name)

    # Also check config.tools
    if config and hasattr(config, "tools"):
        for tool in config.tools:
            spec = _extract_tool_spec(tool)
            if spec.name not in [t.name for t in tools]:
                tools.append(spec)

    # Validators
    validators = []
    validator_obj = getattr(agent, "validator", None)
    if validator_obj and callable(validator_obj):
        validators.append(_extract_validator_spec(validator_obj))

    # Extra info
    extra = {}
    if hasattr(agent, "target"):
        extra["target"] = agent.target

    return AgentSpec(
        name=agent_name,
        description=description,
        instructions=instructions,
        tools=tools,
        validators=validators,
        model=model,
        max_retries=max_retries,
        human_in_loop=human_in_loop,
        source_class=agent.__class__.__name__,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Claude Code Skills / Agent Skills Open Standard
# ---------------------------------------------------------------------------

# Mapping from aw tool names to Claude Code built-in tools
_AW_TOOL_TO_CLAUDE_TOOL = {
    "CodeInterpreterTool": "Bash",
    "SafeCodeInterpreter": "Bash",
    "FileSamplerTool": "Read",
}


def _spec_to_allowed_tools(spec: AgentSpec) -> list[str]:
    """Map aw tools to Claude Code allowed-tools."""
    claude_tools = set()
    for tool in spec.tools:
        mapped = _AW_TOOL_TO_CLAUDE_TOOL.get(tool.name)
        if mapped:
            claude_tools.add(mapped)
    # Always include Read for data work
    claude_tools.add("Read")
    return sorted(claude_tools)


def _spec_to_skill_instructions(spec: AgentSpec) -> str:
    """Render the markdown body of a SKILL.md from an AgentSpec."""
    sections = []

    # Instructions (which already contains the class docstring as first part)
    if spec.instructions:
        sections.append(f"## Instructions\n\n{spec.instructions}")
    elif spec.description:
        sections.append(f"## Purpose\n\n{spec.description}")

    # Tools section
    if spec.tools:
        tool_lines = []
        for tool in spec.tools:
            line = f"- **{tool.name}**"
            if tool.description:
                # First sentence only
                first_sentence = tool.description.split(".")[0]
                line += f": {first_sentence}"
            tool_lines.append(line)
        sections.append("## Available Tools\n\n" + "\n".join(tool_lines))

    # Validation section
    if spec.validators:
        val_lines = []
        for v in spec.validators:
            if v.checks:
                for check in v.checks:
                    val_lines.append(f"- {check}")
            elif v.description:
                val_lines.append(f"- {v.description.split(chr(10))[0]}")
            else:
                val_lines.append(f"- {v.name}")
        sections.append(
            "## Validation\n\nThe output must pass these checks:\n\n"
            + "\n".join(val_lines)
        )

    # Retry behavior
    if spec.max_retries > 1:
        sections.append(
            f"## Error Recovery\n\n"
            f"If a step fails, retry with adjusted parameters "
            f"(up to {spec.max_retries} attempts).\n"
            f"Analyze the error message to determine what to change."
        )

    # Target (if present)
    target = spec.extra.get("target")
    if target:
        sections.append(f"## Target\n\nPrepare data for target format: **{target}**")

    return "\n\n".join(sections)


def to_claude_skill(
    agent: Any,
    name: str = None,
    description: str = None,
    extra_tools: list = None,
    disable_model_invocation: bool = False,
) -> str:
    """Translate an aw agent to a Claude Code SKILL.md string.

    Generates a complete SKILL.md file with YAML frontmatter and
    markdown instructions, compatible with both Claude Code skills
    and the Agent Skills Open Standard (agentskills.io).

    Args:
        agent: An aw agent instance
        name: Override skill name (defaults to kebab-case of class name)
        description: Override description
        extra_tools: Additional Claude Code tools to allow
        disable_model_invocation: If True, skill is manual-only (/name)

    Returns:
        Complete SKILL.md content as a string

    Example:
        >>> from aw import LoadingAgent
        >>> skill = to_claude_skill(LoadingAgent(), name='data-loading')
        >>> '---' in skill
        True
        >>> 'data-loading' in skill
        True
    """
    spec = extract_agent_spec(agent, name=name)

    # Derive skill name (kebab-case)
    skill_name = name or _to_kebab_case(spec.name)

    # Description
    skill_description = description or spec.description

    # Allowed tools
    allowed_tools = _spec_to_allowed_tools(spec)
    if extra_tools:
        allowed_tools = sorted(set(allowed_tools) | set(extra_tools))

    # Build frontmatter
    fm_lines = ["---"]
    fm_lines.append(f"name: {skill_name}")
    if skill_description:
        # Escape for YAML
        fm_lines.append(f"description: >-")
        fm_lines.append(f"  {skill_description}")
    if allowed_tools:
        fm_lines.append(f"allowed-tools: {' '.join(allowed_tools)}")
    if spec.model:
        fm_lines.append(f"model: {spec.model}")
    if disable_model_invocation:
        fm_lines.append("disable-model-invocation: true")
    fm_lines.append("---")

    frontmatter = "\n".join(fm_lines)

    # Build body
    body = _spec_to_skill_instructions(spec)

    return f"{frontmatter}\n\n{body}\n"


def write_skill_directory(
    agent: Any,
    output_dir: Union[str, Path],
    name: str = None,
    description: str = None,
    include_scripts: bool = True,
) -> Path:
    """Write a complete Claude Code skill directory.

    Creates:
        output_dir/
        ├── SKILL.md
        └── scripts/          (if include_scripts and agent has tools)
            └── validate.py   (validator wrapper script)

    Args:
        agent: An aw agent instance
        output_dir: Directory to write the skill to
        name: Override skill name
        description: Override description
        include_scripts: Whether to generate helper scripts

    Returns:
        Path to the created directory

    Example:
        >>> from aw import LoadingAgent
        >>> path = write_skill_directory(  # doctest: +SKIP
        ...     LoadingAgent(), '/tmp/test-skill', name='data-loading'
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write SKILL.md
    skill_md = to_claude_skill(agent, name=name, description=description)
    (output_dir / "SKILL.md").write_text(skill_md)

    # Write helper scripts
    if include_scripts:
        spec = extract_agent_spec(agent, name=name)
        scripts_dir = output_dir / "scripts"

        # Generate validator script if we have validators
        if spec.validators:
            scripts_dir.mkdir(exist_ok=True)
            validator_script = _generate_validator_script(spec)
            (scripts_dir / "validate.py").write_text(validator_script)

    return output_dir


def _generate_validator_script(spec: AgentSpec) -> str:
    """Generate a Python validation script from validator specs."""
    lines = [
        "#!/usr/bin/env python3",
        '"""Auto-generated validation script from aw agent spec.',
        "",
        f"Source agent: {spec.source_class}",
        '"""',
        "",
        "import sys",
        "import json",
        "",
        "",
        "def validate(artifact):",
        '    """Validate the artifact meets requirements."""',
        "    errors = []",
        "",
    ]

    for v in spec.validators:
        for check in v.checks:
            lines.append(f"    # Check: {check}")
        if v.description:
            lines.append(f"    # {v.description.split(chr(10))[0]}")

    lines.extend(
        [
            "    if artifact is None:",
            "        errors.append('Artifact is None')",
            '    elif hasattr(artifact, "__len__") and len(artifact) == 0:',
            "        errors.append('Artifact is empty')",
            "",
            "    return len(errors) == 0, errors",
            "",
            "",
            'if __name__ == "__main__":',
            "    # Read artifact from stdin or file argument",
            "    if len(sys.argv) > 1:",
            "        import pandas as pd",
            "        artifact = pd.read_csv(sys.argv[1])",
            "    else:",
            "        artifact = json.load(sys.stdin)",
            "",
            "    success, errors = validate(artifact)",
            "    result = {'success': success, 'errors': errors}",
            "    print(json.dumps(result, indent=2))",
            "    sys.exit(0 if success else 1)",
        ]
    )

    return "\n".join(lines) + "\n"


def workflow_to_skills(
    workflow: Any,
    output_dir: Union[str, Path],
) -> list:
    """Translate an AgenticWorkflow into a set of Claude Code skills.

    Each step in the workflow becomes a separate skill directory.

    Args:
        workflow: An AgenticWorkflow instance
        output_dir: Parent directory for all skill directories

    Returns:
        List of Paths to created skill directories

    Example:
        >>> from aw import create_cosmo_prep_workflow
        >>> workflow = create_cosmo_prep_workflow()
        >>> paths = workflow_to_skills(workflow, '/tmp/skills')  # doctest: +SKIP
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for step_name, step_agent in workflow.steps:
        skill_name = _to_kebab_case(step_name)
        skill_dir = output_dir / skill_name
        path = write_skill_directory(step_agent, skill_dir, name=skill_name)
        paths.append(path)

    return paths


# ---------------------------------------------------------------------------
# CrewAI Format
# ---------------------------------------------------------------------------


def to_crewai_yaml(
    agent: Any,
    name: str = None,
    role: str = None,
    goal: str = None,
    backstory: str = None,
) -> dict:
    """Translate an aw agent to a CrewAI agent YAML config dict.

    CrewAI agents are defined with role, goal, backstory, and tools.
    This function maps aw's AgentSpec to that structure.

    Args:
        agent: An aw agent instance
        name: Override agent name
        role: Override role (defaults to agent description)
        goal: Override goal
        backstory: Override backstory

    Returns:
        Dict suitable for YAML serialization as a CrewAI agent config

    Example:
        >>> from aw import LoadingAgent
        >>> config = to_crewai_yaml(LoadingAgent(), name='data_loader')
        >>> config['role']  # doctest: +SKIP
        'Data Loading Specialist'
    """
    spec = extract_agent_spec(agent, name=name)

    # Derive role from class name or description
    if role is None:
        role = _class_name_to_role(spec.source_class)

    # Derive goal from description
    if goal is None:
        goal = spec.description or f"Execute the {spec.name} task successfully"

    # Derive backstory from instructions
    if backstory is None:
        backstory = (
            f"You are a specialized agent that follows the ReAct pattern "
            f"(Reason-Act-Observe-Validate). "
        )
        if spec.instructions:
            # Use first paragraph of instructions
            first_para = spec.instructions.split("\n\n")[0]
            backstory += first_para

    agent_config = {
        _to_snake_case(name or spec.name): {
            "role": role,
            "goal": goal,
            "backstory": backstory,
            "verbose": True,
            "max_iter": spec.max_retries,
            "tools": [tool.name for tool in spec.tools],
        }
    }

    return agent_config


def workflow_to_crewai_yaml(workflow: Any) -> dict:
    """Translate an AgenticWorkflow to CrewAI agents.yaml + tasks.yaml.

    Args:
        workflow: An AgenticWorkflow instance

    Returns:
        Dict with 'agents' and 'tasks' keys, each containing
        YAML-serializable config dicts

    Example:
        >>> from aw import create_cosmo_prep_workflow
        >>> workflow = create_cosmo_prep_workflow()
        >>> config = workflow_to_crewai_yaml(workflow)
        >>> 'agents' in config and 'tasks' in config
        True
    """
    agents = {}
    tasks = {}

    prev_step_name = None
    for step_name, step_agent in workflow.steps:
        snake_name = _to_snake_case(step_name)

        # Agent config
        agent_config = to_crewai_yaml(step_agent, name=step_name)
        agents.update(agent_config)

        # Task config
        spec = extract_agent_spec(step_agent, name=step_name)
        task = {
            "description": spec.description or f"Execute {step_name}",
            "expected_output": f"Validated output from {step_name} step",
            "agent": snake_name,
        }
        if prev_step_name:
            task["context"] = [_to_snake_case(prev_step_name)]

        tasks[f"{snake_name}_task"] = task
        prev_step_name = step_name

    return {"agents": agents, "tasks": tasks}


# ---------------------------------------------------------------------------
# OpenAI Tool Schemas
# ---------------------------------------------------------------------------


def to_openai_tools(agent: Any) -> list:
    """Translate an aw agent's tools to OpenAI function-calling tool schemas.

    Generates JSON-Schema-based tool definitions compatible with the
    OpenAI Chat Completions API and Responses API.

    Args:
        agent: An aw agent instance

    Returns:
        List of tool definition dicts in OpenAI format

    Example:
        >>> from aw import LoadingAgent
        >>> tools = to_openai_tools(LoadingAgent())
        >>> all(t['type'] == 'function' for t in tools)
        True
    """
    spec = extract_agent_spec(agent)
    openai_tools = []

    for tool in spec.tools:
        # Build parameter schema from extracted info
        properties = {}
        required = []

        for pname, pinfo in tool.parameters.items():
            prop = {"type": "string", "description": f"Parameter: {pname}"}
            properties[pname] = prop
            if pinfo.get("default") is None:
                required.append(pname)

        tool_def = {
            "type": "function",
            "function": {
                "name": _to_snake_case(tool.name),
                "description": tool.description or f"Tool: {tool.name}",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        openai_tools.append(tool_def)

    return openai_tools


def to_openai_assistant(
    agent: Any,
    name: str = None,
    model: str = "gpt-4",
) -> dict:
    """Translate an aw agent to an OpenAI Assistant-style config dict.

    Generates a configuration suitable for creating an OpenAI Assistant
    (or Responses API agent) via the API.

    Args:
        agent: An aw agent instance
        name: Override name
        model: Override model

    Returns:
        Dict with assistant configuration

    Example:
        >>> from aw import LoadingAgent
        >>> config = to_openai_assistant(LoadingAgent())
        >>> 'instructions' in config
        True
    """
    spec = extract_agent_spec(agent, name=name)
    tools = to_openai_tools(agent)

    # Always include code_interpreter for aw agents (they execute code)
    has_code_tool = any(
        t.name in ("CodeInterpreterTool", "SafeCodeInterpreter") for t in spec.tools
    )
    if has_code_tool:
        tools.append({"type": "code_interpreter"})

    return {
        "name": name or spec.name,
        "instructions": spec.instructions or spec.description,
        "model": model if model != "gpt-4" else (spec.model or model),
        "tools": tools,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_kebab_case(name: str) -> str:
    """Convert CamelCase or snake_case to kebab-case.

    >>> _to_kebab_case('LoadingAgent')
    'loading-agent'
    >>> _to_kebab_case('cosmo_ready')
    'cosmo-ready'
    >>> _to_kebab_case('already-kebab')
    'already-kebab'
    """
    import re

    # CamelCase to separated
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"-\1", name)
    # Underscores to hyphens
    s = s.replace("_", "-")
    return s.lower()


def _to_snake_case(name: str) -> str:
    """Convert CamelCase or kebab-case to snake_case.

    >>> _to_snake_case('LoadingAgent')
    'loading_agent'
    >>> _to_snake_case('data-loading')
    'data_loading'
    """
    import re

    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name)
    s = s.replace("-", "_")
    return s.lower()


def _class_name_to_role(class_name: str) -> str:
    """Convert a class name to a human-readable role.

    >>> _class_name_to_role('LoadingAgent')
    'Data Loading Specialist'
    >>> _class_name_to_role('PreparationAgent')
    'Data Preparation Specialist'
    """
    import re

    # Remove 'Agent' suffix
    name = re.sub(r"Agent$", "", class_name)
    # Split CamelCase
    words = re.sub(r"(?<=[a-z])([A-Z])", r" \1", name).split()

    if not words:
        return "Specialist"

    # Add context-appropriate prefix/suffix
    role_words = []
    if words[0].lower() not in ("data", "file", "text"):
        role_words.append("Data")
    role_words.extend(words)
    role_words.append("Specialist")

    return " ".join(role_words)
