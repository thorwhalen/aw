# aw.translators

Translators from aw agent specifications to other agent formats.

Supports:

- Claude Code Skills / Agent Skills Open Standard (SKILL.md + scripts/)
- CrewAI (YAML agent/task configs)
- OpenAI tool schemas (JSON)

The core pattern is:

1. extract_agent_spec() introspects an aw agent into a normalized dict
2. Format-specific renderers convert the spec to target formats

### Example

```pycon
>>> from aw import LoadingAgent
>>> from aw.translators import to_claude_skill
>>> agent = LoadingAgent()
>>> skill_md = to_claude_skill(agent, name='data-loading')
>>> print(skill_md)
```

```pycon
>>> # Write a complete skill directory
>>> from aw.translators import write_skill_directory
>>> write_skill_directory(agent, '/path/to/skills/data-loading')
```

### Functions

| [`claude_skill_from_spec`](#aw.translators.claude_skill_from_spec)(spec, \*[, name, ...])      | Render a Claude Code SKILL.md string from an [`AgentSpec`](#aw.translators.AgentSpec).         |
|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [`crewai_yaml_from_spec`](#aw.translators.crewai_yaml_from_spec)(spec, \*[, name, role, ...]) | Render a CrewAI agent config dict from an [`AgentSpec`](#aw.translators.AgentSpec).            |
| [`extract_agent_spec`](#aw.translators.extract_agent_spec)(agent[, name])                  | Extract a normalized AgentSpec from any aw agent.                                                                |
| [`openai_assistant_from_spec`](#aw.translators.openai_assistant_from_spec)(spec, \*[, name, ...])  | Render an OpenAI Assistant-style config dict from an [`AgentSpec`](#aw.translators.AgentSpec). |
| [`openai_tools_from_spec`](#aw.translators.openai_tools_from_spec)(spec)                       | Render OpenAI function-calling tool schemas from an [`AgentSpec`](#aw.translators.AgentSpec).  |
| [`to_claude_skill`](#aw.translators.to_claude_skill)(agent[, name, description, ...])   | Translate an aw agent to a Claude Code SKILL.md string.                                                          |
| [`to_crewai_yaml`](#aw.translators.to_crewai_yaml)(agent[, name, role, goal, ...])     | Translate an aw agent to a CrewAI agent YAML config dict.                                                        |
| [`to_openai_assistant`](#aw.translators.to_openai_assistant)(agent[, name, model])          | Translate an aw agent to an OpenAI Assistant-style config dict.                                                  |
| [`to_openai_tools`](#aw.translators.to_openai_tools)(agent)                             | Translate an aw agent's tools to OpenAI function-calling tool schemas.                                           |
| [`workflow_to_crewai_yaml`](#aw.translators.workflow_to_crewai_yaml)(workflow)                  | Translate an AgenticWorkflow to CrewAI agents.yaml + tasks.yaml.                                                 |
| [`workflow_to_skills`](#aw.translators.workflow_to_skills)(workflow, output_dir)           | Translate an AgenticWorkflow into a set of Claude Code skills.                                                   |
| [`write_skill_directory`](#aw.translators.write_skill_directory)(agent, output_dir[, ...])    | Write a complete Claude Code skill directory.                                                                    |

### Classes

| [`AgentSpec`](#aw.translators.AgentSpec)(name[, description, instructions, ...])   | Normalized, format-agnostic description of an aw agent.   |
|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| [`ToolSpec`](#aw.translators.ToolSpec)(name[, description, parameters, source])   | Normalized description of a tool.                         |
| [`ValidatorSpec`](#aw.translators.ValidatorSpec)(name[, description, checks])          | Normalized description of a validator.                    |

### *class* aw.translators.AgentSpec(name, description='', instructions='', tools=list(), validators=list(), model='', max_retries=3, human_in_loop=False, source_class='', extra=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Normalized, format-agnostic description of an aw agent.

This is the intermediate representation that all translators consume.
It captures the essential components of an agent without being tied
to any specific framework.

### Example

```pycon
>>> from aw import LoadingAgent
>>> spec = extract_agent_spec(LoadingAgent())
>>> spec.name
'LoadingAgent'
```

### *class* aw.translators.ToolSpec(name, description='', parameters=dict(), source='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Normalized description of a tool.

### *class* aw.translators.ValidatorSpec(name, description='', checks=list())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Normalized description of a validator.

### aw.translators.claude_skill_from_spec(spec, , name=None, description=None, extra_tools=None, disable_model_invocation=False)

Render a Claude Code SKILL.md string from an [`AgentSpec`](#aw.translators.AgentSpec).

The rendering core of [`to_claude_skill()`](#aw.translators.to_claude_skill) — driveable by any `AgentSpec`
(not only a live aw agent), so other tools (e.g. `coact`) can reuse it.

### Example

```pycon
>>> from aw import LoadingAgent
>>> from aw.translators import extract_agent_spec, claude_skill_from_spec
>>> spec = extract_agent_spec(LoadingAgent(), name='data-loading')
>>> '---' in claude_skill_from_spec(spec)
True
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### aw.translators.crewai_yaml_from_spec(spec, , name=None, role=None, goal=None, backstory=None)

Render a CrewAI agent config dict from an [`AgentSpec`](#aw.translators.AgentSpec).

The rendering core of [`to_crewai_yaml()`](#aw.translators.to_crewai_yaml), driveable by any `AgentSpec`.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### aw.translators.extract_agent_spec(agent, name=None)

Extract a normalized AgentSpec from any aw agent.

Introspects the agent’s class, config, tools, and validators
to build a format-agnostic intermediate representation.

* **Parameters:**
  * **agent** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – An aw agent instance (LoadingAgent, PreparationAgent, etc.)
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Override name (defaults to class name)
* **Return type:**
  [`AgentSpec`](#aw.translators.AgentSpec)
* **Returns:**
  AgentSpec with all extractable information

### Example

```pycon
>>> from aw import LoadingAgent
>>> spec = extract_agent_spec(LoadingAgent())
>>> spec.name
'LoadingAgent'
>>> len(spec.tools) >= 1
True
```

### aw.translators.openai_assistant_from_spec(spec, , name=None, model='gpt-4')

Render an OpenAI Assistant-style config dict from an [`AgentSpec`](#aw.translators.AgentSpec).

The rendering core of [`to_openai_assistant()`](#aw.translators.to_openai_assistant), driveable by any `AgentSpec`.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### aw.translators.openai_tools_from_spec(spec)

Render OpenAI function-calling tool schemas from an [`AgentSpec`](#aw.translators.AgentSpec).

The rendering core of [`to_openai_tools()`](#aw.translators.to_openai_tools), driveable by any `AgentSpec`.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

### aw.translators.to_claude_skill(agent, name=None, description=None, extra_tools=None, disable_model_invocation=False)

Translate an aw agent to a Claude Code SKILL.md string.

Generates a complete SKILL.md file with YAML frontmatter and
markdown instructions, compatible with both Claude Code skills
and the Agent Skills Open Standard (agentskills.io).

* **Parameters:**
  * **agent** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – An aw agent instance
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Override skill name (defaults to kebab-case of class name)
  * **description** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Override description
  * **extra_tools** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)) – Additional Claude Code tools to allow
  * **disable_model_invocation** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, skill is manual-only (/name)
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Complete SKILL.md content as a string

### Example

```pycon
>>> from aw import LoadingAgent
>>> skill = to_claude_skill(LoadingAgent(), name='data-loading')
>>> '---' in skill
True
>>> 'data-loading' in skill
True
```

### aw.translators.to_crewai_yaml(agent, name=None, role=None, goal=None, backstory=None)

Translate an aw agent to a CrewAI agent YAML config dict.

CrewAI agents are defined with role, goal, backstory, and tools.
This function maps aw’s AgentSpec to that structure.

* **Parameters:**
  * **agent** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – An aw agent instance
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Override agent name
  * **role** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Override role (defaults to agent description)
  * **goal** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Override goal
  * **backstory** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Override backstory
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dict suitable for YAML serialization as a CrewAI agent config

### Example

```pycon
>>> from aw import LoadingAgent
>>> config = to_crewai_yaml(LoadingAgent(), name='data_loader')
>>> config['role']
'Data Loading Specialist'
```

### aw.translators.to_openai_assistant(agent, name=None, model='gpt-4')

Translate an aw agent to an OpenAI Assistant-style config dict.

Generates a configuration suitable for creating an OpenAI Assistant
(or Responses API agent) via the API.

* **Parameters:**
  * **agent** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – An aw agent instance
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Override name
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Override model
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dict with assistant configuration

### Example

```pycon
>>> from aw import LoadingAgent
>>> config = to_openai_assistant(LoadingAgent())
>>> 'instructions' in config
True
```

### aw.translators.to_openai_tools(agent)

Translate an aw agent’s tools to OpenAI function-calling tool schemas.

Generates JSON-Schema-based tool definitions compatible with the
OpenAI Chat Completions API and Responses API.

* **Parameters:**
  **agent** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – An aw agent instance
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)
* **Returns:**
  List of tool definition dicts in OpenAI format

### Example

```pycon
>>> from aw import LoadingAgent
>>> tools = to_openai_tools(LoadingAgent())
>>> all(t['type'] == 'function' for t in tools)
True
```

### aw.translators.workflow_to_crewai_yaml(workflow)

Translate an AgenticWorkflow to CrewAI agents.yaml + tasks.yaml.

* **Parameters:**
  **workflow** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – An AgenticWorkflow instance
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dict with ‘agents’ and ‘tasks’ keys, each containing
  YAML-serializable config dicts

### Example

```pycon
>>> from aw import create_cosmo_prep_workflow
>>> workflow = create_cosmo_prep_workflow()
>>> config = workflow_to_crewai_yaml(workflow)
>>> 'agents' in config and 'tasks' in config
True
```

### aw.translators.workflow_to_skills(workflow, output_dir)

Translate an AgenticWorkflow into a set of Claude Code skills.

Each step in the workflow becomes a separate skill directory.

* **Parameters:**
  * **workflow** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – An AgenticWorkflow instance
  * **output_dir** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Parent directory for all skill directories
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)
* **Returns:**
  List of Paths to created skill directories

### Example

```pycon
>>> from aw import create_cosmo_prep_workflow
>>> workflow = create_cosmo_prep_workflow()
>>> paths = workflow_to_skills(workflow, '/tmp/skills')
```

### aw.translators.write_skill_directory(agent, output_dir, name=None, description=None, include_scripts=True)

Write a complete Claude Code skill directory.

Creates:

```default
output_dir/
├── SKILL.md
└── scripts/          (if include_scripts and agent has tools)
    └── validate.py   (validator wrapper script)
```

* **Parameters:**
  * **agent** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – An aw agent instance
  * **output_dir** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Directory to write the skill to
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Override skill name
  * **description** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Override description
  * **include_scripts** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to generate helper scripts
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
* **Returns:**
  Path to the created directory

### Example

```pycon
>>> from aw import LoadingAgent
>>> path = write_skill_directory(
...     LoadingAgent(), '/tmp/test-skill', name='data-loading'
... )
```
