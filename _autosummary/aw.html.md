# aw

aw - Agentic Workflows for Data Preparation

An AI agent package for data preparation with a focus on loading and preparing
data for various purposes (e.g., visualization with cosmograph).

Key Features:

- AgenticStep protocol for building modular agents
- ReAct pattern (Reason-Act-Observe) with retry logic
- Three validation flavors: schema, info-dict, and functional
- Code execution with safe defaults and extensibility
- Loading and Preparation agents for data workflows
- Cosmograph-specific validators and utilities
- Orchestration for chaining multiple steps

### Example

```pycon
>>> from aw import load_for_cosmo
>>> df, metadata = load_for_cosmo('data.csv')
>>> from cosmograph import cosmo
>>> params = metadata['preparing']['metadata']['validation_result']['params']
>>> cosmo(df, **params)
```

Architecture:

> - aw.base: Core protocols and configurations
> - aw.validation: Three validation flavors
> - aw.tools: Code execution and agent tools
> - aw.utils: Helper functions and facades
> - aw.loading: LoadingAgent for data ingestion
> - aw.preparing: PreparationAgent for data transformation
> - aw.cosmo: Cosmograph-specific validators
> - aw.orchestration: Workflow management

### Functions

| [`schema_validator`](#aw.schema_validator)(schema)                           | Create a validator from a schema (Pydantic, JSON Schema, etc.).                                                  |
|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [`info_dict_validator`](#aw.info_dict_validator)(compute_info, check_info)      | Create a validator that computes info then checks it.                                                            |
| [`functional_validator`](#aw.functional_validator)(try_function[, ...])          | Create a validator that tries to use the artifact for its purpose.                                               |
| [`all_validators`](#aw.all_validators)(\*validators)                       | Combine multiple validators - all must pass.                                                                     |
| [`any_validator`](#aw.any_validator)(\*validators)                        | Combine multiple validators - at least one must pass.                                                            |
| [`is_type`](#aw.is_type)(expected_type)                             | Validator that checks artifact type.                                                                             |
| [`is_not_empty`](#aw.is_not_empty)()                                     | Validator that checks artifact is not empty.                                                                     |
| [`has_attributes`](#aw.has_attributes)(\*\*required_attrs)                 | Validator that checks artifact has required attributes.                                                          |
| [`create_langchain_executor`](#aw.create_langchain_executor)()                        | Create an executor using LangChain's CodeInterpreter (if available).                                             |
| [`djoin`](#aw.djoin)(\*p)                                         | Join two or more pathname components, inserting '/' as needed.                                                   |
| [`infer_loader_from_extension`](#aw.infer_loader_from_extension)(extension)             | Infer pandas loader function from file extension.                                                                |
| [`infer_loader_params`](#aw.infer_loader_params)(extension[, sample_text])      | Infer parameters for pandas loader based on file characteristics.                                                |
| [`compute_dataframe_info`](#aw.compute_dataframe_info)(df)                         | Compute comprehensive info about a DataFrame.                                                                    |
| [`get_numeric_columns`](#aw.get_numeric_columns)(df[, exclude_nulls])           | Get numeric columns from DataFrame.                                                                              |
| [`default_llm_factory`](#aw.default_llm_factory)([model])                       | Factory to create LLM function from various specifications.                                                      |
| [`create_openai_chat`](#aw.create_openai_chat)([model])                        | Create a text-to-text chat function using OpenAI API.                                                            |
| [`create_oa_chat`](#aw.create_oa_chat)()                                   | Create a chat function using the oa package.                                                                     |
| [`create_loading_agent`](#aw.create_loading_agent)([llm, validator, ...])        | Factory function to create a loading agent.                                                                      |
| [`create_preparation_agent`](#aw.create_preparation_agent)([target, ...])            | Factory function to create a preparation agent.                                                                  |
| [`create_cosmo_validator`](#aw.create_cosmo_validator)([cosmo_function, ...])      | Create a validator for cosmograph requirements.                                                                  |
| [`try_cosmo_visualization`](#aw.try_cosmo_visualization)(df[, cosmo_function])      | Try to create a cosmograph visualization.                                                                        |
| [`basic_cosmo_validator`](#aw.basic_cosmo_validator)()                            | Basic validator that checks structural requirements only.                                                        |
| [`strict_cosmo_validator`](#aw.strict_cosmo_validator)()                           | Strict validator that actually calls cosmograph.                                                                 |
| [`infer_cosmo_params`](#aw.infer_cosmo_params)(df)                             | Infer suitable cosmograph parameters from DataFrame.                                                             |
| [`create_data_prep_workflow`](#aw.create_data_prep_workflow)([loading_config, ...])   | Factory to create a data preparation workflow.                                                                   |
| [`create_cosmo_prep_workflow`](#aw.create_cosmo_prep_workflow)([...])                  | Create a workflow specifically for cosmograph preparation.                                                       |
| [`load_and_prepare`](#aw.load_and_prepare)(source_uri[, target, ...])        | Convenience function to load and prepare data in one call.                                                       |
| [`load_for_cosmo`](#aw.load_for_cosmo)(source_uri[, max_retries, strict])  | Load and prepare data specifically for cosmograph visualization.                                                 |
| [`extract_agent_spec`](#aw.extract_agent_spec)(agent[, name])                  | Extract a normalized AgentSpec from any aw agent.                                                                |
| [`to_claude_skill`](#aw.to_claude_skill)(agent[, name, description, ...])   | Translate an aw agent to a Claude Code SKILL.md string.                                                          |
| [`claude_skill_from_spec`](#aw.claude_skill_from_spec)(spec, \*[, name, ...])      | Render a Claude Code SKILL.md string from an [`AgentSpec`](#aw.AgentSpec).         |
| [`write_skill_directory`](#aw.write_skill_directory)(agent, output_dir[, ...])    | Write a complete Claude Code skill directory.                                                                    |
| [`workflow_to_skills`](#aw.workflow_to_skills)(workflow, output_dir)           | Translate an AgenticWorkflow into a set of Claude Code skills.                                                   |
| [`to_crewai_yaml`](#aw.to_crewai_yaml)(agent[, name, role, goal, ...])     | Translate an aw agent to a CrewAI agent YAML config dict.                                                        |
| [`crewai_yaml_from_spec`](#aw.crewai_yaml_from_spec)(spec, \*[, name, role, ...]) | Render a CrewAI agent config dict from an [`AgentSpec`](#aw.AgentSpec).            |
| [`workflow_to_crewai_yaml`](#aw.workflow_to_crewai_yaml)(workflow)                  | Translate an AgenticWorkflow to CrewAI agents.yaml + tasks.yaml.                                                 |
| [`to_openai_tools`](#aw.to_openai_tools)(agent)                             | Translate an aw agent's tools to OpenAI function-calling tool schemas.                                           |
| [`openai_tools_from_spec`](#aw.openai_tools_from_spec)(spec)                       | Render OpenAI function-calling tool schemas from an [`AgentSpec`](#aw.AgentSpec).  |
| [`to_openai_assistant`](#aw.to_openai_assistant)(agent[, name, model])          | Translate an aw agent to an OpenAI Assistant-style config dict.                                                  |
| [`openai_assistant_from_spec`](#aw.openai_assistant_from_spec)(spec, \*[, name, ...])  | Render an OpenAI Assistant-style config dict from an [`AgentSpec`](#aw.AgentSpec). |

### Classes

| [`AgenticStep`](#aw.AgenticStep)(\*args, \*\*kwargs)                   | Protocol defining the interface for any agentic step in a workflow.   |
|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| [`Validator`](#aw.Validator)(\*args, \*\*kwargs)                     | Protocol for validation functions.                                    |
| [`Tool`](#aw.Tool)(\*args, \*\*kwargs)                          | Protocol for tools that agents can use.                               |
| [`StepConfig`](#aw.StepConfig)([llm, validator, tools, ...])          | Configuration for an AgenticStep.                                     |
| [`GlobalConfig`](#aw.GlobalConfig)([llm, max_retries, human_in_loop])   | Global configuration with cascading defaults.                         |
| [`Context`](#aw.Context)([initial_data])                           | Context for sharing state and artifacts between agentic steps.        |
| [`CodeInterpreterTool`](#aw.CodeInterpreterTool)([allowed_modules, ...])       | Tool for executing Python code in a controlled environment.           |
| [`SafeCodeInterpreter`](#aw.SafeCodeInterpreter)(\*\*kwargs)                   | Extra-safe code interpreter with restricted operations.               |
| [`ExecutionResult`](#aw.ExecutionResult)(success[, output, error, ...])    | Result of code execution.                                             |
| [`FileSamplerTool`](#aw.FileSamplerTool)([sample_size])                    | Tool to sample and analyze file metadata and content.                 |
| [`LoadingAgent`](#aw.LoadingAgent)([config])                            | Agent that loads data from various sources into pandas DataFrames.    |
| [`PreparationAgent`](#aw.PreparationAgent)([config, target, ...])           | Agent that prepares data to meet target requirements.                 |
| [`AgenticWorkflow`](#aw.AgenticWorkflow)([context])                        | Orchestrates a chain of agentic steps.                                |
| [`InteractiveWorkflow`](#aw.InteractiveWorkflow)([context])                    | Workflow with human-in-the-loop capabilities.                         |
| [`AgentSpec`](#aw.AgentSpec)(name[, description, instructions, ...]) | Normalized, format-agnostic description of an aw agent.               |
| [`ToolSpec`](#aw.ToolSpec)(name[, description, parameters, source]) | Normalized description of a tool.                                     |
| [`ValidatorSpec`](#aw.ValidatorSpec)(name[, description, checks])        | Normalized description of a validator.                                |

### *class* aw.AgentSpec(name, description='', instructions='', tools=list(), validators=list(), model='', max_retries=3, human_in_loop=False, source_class='', extra=dict())

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

### *class* aw.AgenticStep(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)[`ArtifactType`]

Protocol defining the interface for any agentic step in a workflow.

Each step follows the ReAct pattern:

1. Reason (Thought): Analyze input and context
2. Act (Action): Generate and execute code or use tools
3. Observe: Capture results and validate
4. Repeat or finish based on validation

### Example

```pycon
>>> class MyAgent:
...     def execute(self, input_data, context):
...         # Agent implementation
...         return result, metadata
```

#### execute(input_data, context)

Execute the agentic step.

* **Parameters:**
  * **input_data** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The input to process
  * **context** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Mutable mapping containing shared state and history
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ArtifactType`, [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]
* **Returns:**
  Tuple of (artifact, metadata) where artifact is the main output
  and metadata contains auxiliary information

### *class* aw.AgenticWorkflow(context=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Orchestrates a chain of agentic steps.

Manages the execution of multiple steps in sequence, handling
context/artifact passing between steps.

### Example

```pycon
>>> workflow = AgenticWorkflow()
>>> workflow.add_step('loading', loading_agent)
>>> workflow.add_step('preparing', preparing_agent)
>>> result = workflow.run(source_uri)
```

#### add_step(name, step)

Add a step to the workflow.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name/identifier for the step
  * **step** ([`AgenticStep`](aw.base.html.md#aw.base.AgenticStep)) – Agent or step implementation
* **Return type:**
  [`AgenticWorkflow`](aw.orchestration.html.md#aw.orchestration.AgenticWorkflow)
* **Returns:**
  Self for chaining

#### run(initial_input)

Execute the workflow.

* **Parameters:**
  **initial_input** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Input to the first step
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]
* **Returns:**
  Tuple of (final_artifact, workflow_metadata)

#### run_partial(initial_input, stop_after=None)

Run workflow up to a specific step.

* **Parameters:**
  * **initial_input** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Input to first step
  * **stop_after** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of step to stop after (runs all if None)
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]
* **Returns:**
  Tuple of (artifact, metadata)

### *class* aw.CodeInterpreterTool(allowed_modules=None, global_context=None, executor=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Tool for executing Python code in a controlled environment.

Provides a safe default implementation using exec() with limited namespace,
and allows injection of more robust backends.

### Example

```pycon
>>> tool = CodeInterpreterTool()
>>> result = tool("x = 5; y = x * 2; print(y)")
>>> result.success
True
>>> result.output.strip()
'10'
```

### *class* aw.Context(initial_data=None)

Bases: [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

Context for sharing state and artifacts between agentic steps.

Implements MutableMapping interface for dict-like behavior while
providing additional functionality for managing agent state.

### Example

```pycon
>>> ctx = Context()
>>> ctx['loading'] = {'df': df, 'info': {...}}
>>> ctx['preparing'] = {'df': prepared_df}
```

#### *property* history *: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]*

Access the history of all context updates.

#### snapshot()

Create a snapshot of current context state.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  A dict copy of current context data

### *class* aw.ExecutionResult(success, output='', error='', traceback_str='', result=None, locals_dict=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Result of code execution.

#### success

Whether execution succeeded without exceptions

#### output

Standard output captured during execution

#### error

Error message if execution failed

#### traceback

Full traceback if execution failed

#### result

The return value or final expression value

#### locals

Local variables after execution

### *class* aw.FileSamplerTool(sample_size=1024)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Tool to sample and analyze file metadata and content.

Helps agents understand what kind of data they’re dealing with
before attempting to load it.

### Example

```pycon
>>> sampler = FileSamplerTool()
>>> info = sampler('/path/to/data.csv')
>>> info['extension']
'.csv'
```

### *class* aw.GlobalConfig(llm='gpt-4', max_retries=3, human_in_loop=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Global configuration with cascading defaults.

Provides defaults that can be overridden at step or agent level.

### Example

```pycon
>>> global_cfg = GlobalConfig(llm="gpt-4", max_retries=5)
>>> step_cfg = global_cfg.override(llm="gpt-3.5-turbo")
```

#### override(\*\*kwargs)

Create a StepConfig with overridden values.

* **Parameters:**
  **\*\*kwargs** – Values to override from global defaults
* **Return type:**
  [`StepConfig`](aw.base.html.md#aw.base.StepConfig)
* **Returns:**
  A new StepConfig with merged configuration

### *class* aw.InteractiveWorkflow(context=None)

Bases: [`AgenticWorkflow`](aw.orchestration.html.md#aw.orchestration.AgenticWorkflow)

Workflow with human-in-the-loop capabilities.

Pauses execution to request human input/approval at configured points.

### Example

```pycon
>>> workflow = InteractiveWorkflow()
>>> workflow.add_step('loading', agent, require_approval=True)
>>> result = workflow.run_interactive(source_uri)
```

#### add_step(name, step, require_approval=False)

Add step with optional approval requirement.

* **Return type:**
  [`InteractiveWorkflow`](aw.orchestration.html.md#aw.orchestration.InteractiveWorkflow)

#### run_interactive(initial_input)

Run workflow with human-in-loop checkpoints.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

#### set_approval_callback(callback)

Set callback for human approval.

* **Parameters:**
  **callback** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – Function(step_name, artifact, metadata) -> bool
  Returns True to continue, False to abort
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* aw.LoadingAgent(config=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Agent that loads data from various sources into pandas DataFrames.

Uses ReAct loop:

1. Thought: Analyze source (extension, sample) to choose loader
2. Action: Generate code to load data
3. Observe: Execute code and capture result/error
4. Validate: Check if result is valid DataFrame
5. Repeat or finish

### Example

```pycon
>>> agent = LoadingAgent()
>>> context = Context()
>>> df, metadata = agent.execute('/path/to/data.csv', context)
```

#### execute(source_uri, context)

Execute loading agent to load data from source.

* **Parameters:**
  * **source_uri** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – URI of data source (file path or URL)
  * **context** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Context for storing intermediate results
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]
* **Returns:**
  Tuple of (dataframe, metadata)

### *class* aw.PreparationAgent(config=None, target='generic', target_validator=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Agent that prepares data to meet target requirements.

Uses ReAct loop with functional validation:

1. Thought: Analyze current data state vs. requirements
2. Action: Generate transformation code
3. Observe: Execute code and capture result
4. Validate: Try to use data for its purpose (e.g., visualization)
5. Repeat or finish

### Example

```pycon
>>> agent = PreparationAgent(target='cosmo-ready')
>>> context = Context({'loading': {'df': df}})
>>> prepared_df, metadata = agent.execute(df, context)
```

#### execute(input_df, context)

Execute preparation agent to transform data.

* **Parameters:**
  * **input_df** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Input DataFrame to prepare
  * **context** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Context for storing intermediate results
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]
* **Returns:**
  Tuple of (prepared_dataframe, metadata)

### *class* aw.SafeCodeInterpreter(\*\*kwargs)

Bases: [`CodeInterpreterTool`](aw.tools.html.md#aw.tools.CodeInterpreterTool)

Extra-safe code interpreter with restricted operations.

Disallows file I/O, network access, and other potentially dangerous operations.

### *class* aw.StepConfig(llm='gpt-4', validator=None, tools=list(), max_retries=3, human_in_loop=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Configuration for an AgenticStep.

Supports both simple objects (strings, ints) and callables for maximum
flexibility with dependency injection.

#### llm

Either a model name string or a text-to-text chat function

* **Type:**
  [`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

#### validator

Validation callable or schema object

* **Type:**
  [`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

#### tools

List of callable tools available to the agent

* **Type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]

#### max_retries

Maximum number of retry attempts

* **Type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

#### human_in_loop

Whether to require human approval/intervention

* **Type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### Example

```pycon
>>> config = StepConfig(
...     llm="gpt-4",
...     validator=lambda x: (True, {}),
...     max_retries=3
... )
```

#### resolve_llm()

Resolve llm to a callable function.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  A callable that takes a prompt string and returns a response string

#### resolve_validator()

Resolve validator to a callable.

* **Return type:**
  [`Validator`](aw.base.html.md#aw.base.Validator)
* **Returns:**
  A callable that validates artifacts

### *class* aw.Tool(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Protocol for tools that agents can use.

Tools are callable objects that perform specific actions like
executing code, sampling files, or calling external APIs.

### Example

```pycon
>>> class FileSampler:
...     def __call__(self, uri):
...         # Sample file logic
...         return {'extension': '.csv', 'sample': '...'}
```

### *class* aw.ToolSpec(name, description='', parameters=dict(), source='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Normalized description of a tool.

### *class* aw.Validator(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Protocol for validation functions.

Validators check if an artifact meets requirements and return
both a success indicator and detailed information.

### Example

```pycon
>>> def is_non_empty_dataframe(df):
...     success = df is not None and len(df) > 0
...     info = {'shape': df.shape if success else None}
...     return success, info
```

### *class* aw.ValidatorSpec(name, description='', checks=list())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Normalized description of a validator.

### aw.all_validators(\*validators)

Combine multiple validators - all must pass.

* **Parameters:**
  **\*validators** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Validator functions to combine
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]
* **Returns:**
  A validator that passes only if all validators pass

### Example

```pycon
>>> validate = all_validators(
...     is_dataframe,
...     has_required_columns(['x', 'y']),
...     has_no_nulls
... )
```

### aw.any_validator(\*validators)

Combine multiple validators - at least one must pass.

* **Parameters:**
  **\*validators** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Validator functions to combine
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]
* **Returns:**
  A validator that passes if any validator passes

### aw.basic_cosmo_validator()

Basic validator that checks structural requirements only.

Does not require cosmograph to be installed.

### Example

```pycon
>>> validator = basic_cosmo_validator()
>>> success, info = validator(df)
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)

### aw.claude_skill_from_spec(spec, , name=None, description=None, extra_tools=None, disable_model_invocation=False)

Render a Claude Code SKILL.md string from an [`AgentSpec`](#aw.AgentSpec).

The rendering core of [`to_claude_skill()`](#aw.to_claude_skill) — driveable by any `AgentSpec`
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

### aw.compute_dataframe_info(df)

Compute comprehensive info about a DataFrame.

* **Parameters:**
  **df** – pandas DataFrame
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dict with shape, dtypes, null counts, sample rows, etc.

### Example

```pycon
>>> import pandas as pd
>>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
>>> info = compute_dataframe_info(df)
>>> info['shape']
(2, 2)
```

### aw.create_cosmo_prep_workflow(cosmo_validator=None, max_retries=3)

Create a workflow specifically for cosmograph preparation.

* **Parameters:**
  * **cosmo_validator** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Custom cosmo validator (uses default if None)
  * **max_retries** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum retries for each step
* **Return type:**
  [`AgenticWorkflow`](aw.orchestration.html.md#aw.orchestration.AgenticWorkflow)
* **Returns:**
  Configured workflow

### Example

```pycon
>>> workflow = create_cosmo_prep_workflow()
>>> prepared_df, metadata = workflow.run('data.csv')
```

### aw.create_cosmo_validator(cosmo_function=None, required_columns=2, allow_generated_params=True)

Create a validator for cosmograph requirements.

This validator attempts to actually call cosmograph.cosmo() with the
DataFrame to see if it’s ready for visualization.

* **Parameters:**
  * **cosmo_function** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – The cosmograph.cosmo function (imported if None)
  * **required_columns** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Minimum number of numeric columns needed
  * **allow_generated_params** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to auto-infer x/y columns
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]
* **Returns:**
  Validator function following (artifact) -> (success, info) protocol

### Example

```pycon
>>> validator = create_cosmo_validator()
>>> success, info = validator(df)
>>> if success:
...     print(f"Can visualize with: {info['params']}")
```

### aw.create_data_prep_workflow(loading_config=None, preparing_config=None, target='generic')

Factory to create a data preparation workflow.

Creates a workflow with LoadingAgent -> PreparationAgent.

* **Parameters:**
  * **loading_config** ([`StepConfig`](aw.base.html.md#aw.base.StepConfig)) – Configuration for loading agent
  * **preparing_config** ([`StepConfig`](aw.base.html.md#aw.base.StepConfig)) – Configuration for preparing agent
  * **target** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Target format for preparation
* **Return type:**
  [`AgenticWorkflow`](aw.orchestration.html.md#aw.orchestration.AgenticWorkflow)
* **Returns:**
  Configured workflow

### Example

```pycon
>>> workflow = create_data_prep_workflow(target='cosmo-ready')
>>> df, metadata = workflow.run('/path/to/data.csv')
```

### aw.create_langchain_executor()

Create an executor using LangChain’s CodeInterpreter (if available).

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)], [`ExecutionResult`](aw.tools.html.md#aw.tools.ExecutionResult)]
* **Returns:**
  Executor function compatible with CodeInterpreterTool

### Example

```pycon
>>> try:
...     executor = create_langchain_executor()
...     tool = CodeInterpreterTool(executor=executor)
... except ImportError:
...     # LangChain not available - use default
...     tool = CodeInterpreterTool()
```

### aw.create_loading_agent(llm=None, validator=None, max_retries=3)

Factory function to create a loading agent.

* **Parameters:**
  * **llm** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – LLM model name or callable
  * **validator** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Custom validator (uses default if None)
  * **max_retries** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum retry attempts
* **Return type:**
  [`LoadingAgent`](aw.loading.html.md#aw.loading.LoadingAgent)
* **Returns:**
  Configured LoadingAgent

### Example

```pycon
>>> agent = create_loading_agent(llm="gpt-4", max_retries=5)
```

### aw.create_oa_chat()

Create a chat function using the oa package.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  A callable that takes a prompt and returns a response

### Example

```pycon
>>> chat = create_oa_chat()
>>> response = chat("Hello!")
```

### aw.create_openai_chat(model='gpt-4', \*\*default_kwargs)

Create a text-to-text chat function using OpenAI API.

* **Parameters:**
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – OpenAI model name
  * **\*\*default_kwargs** – Default parameters for API calls
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  A callable that takes a prompt and returns a response

### Example

```pycon
>>> chat = create_openai_chat("gpt-4")
>>> response = chat("What is 2+2?")
```

### aw.create_preparation_agent(target='generic', validator=None, llm=None, max_retries=3)

Factory function to create a preparation agent.

* **Parameters:**
  * **target** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Target format/purpose
  * **validator** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Custom validator function
  * **llm** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – LLM model name or callable
  * **max_retries** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum retry attempts
* **Return type:**
  [`PreparationAgent`](aw.preparing.html.md#aw.preparing.PreparationAgent)
* **Returns:**
  Configured PreparationAgent

### Example

```pycon
>>> agent = create_preparation_agent(
...     target='cosmo-ready',
...     validator=cosmo_validator,
...     max_retries=5
... )
```

### aw.crewai_yaml_from_spec(spec, , name=None, role=None, goal=None, backstory=None)

Render a CrewAI agent config dict from an [`AgentSpec`](#aw.AgentSpec).

The rendering core of [`to_crewai_yaml()`](#aw.to_crewai_yaml), driveable by any `AgentSpec`.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### aw.default_llm_factory(model='gpt-4')

Factory to create LLM function from various specifications.

* **Parameters:**
  **model** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Either a model name string, or a callable
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  A callable text-to-text function

### Example

```pycon
>>> llm = default_llm_factory("gpt-4")
>>> llm = default_llm_factory(lambda p: f"Echo: {p}")
```

### aw.djoin(\*p)

Join two or more pathname components, inserting ‘/’ as needed.
If any component is an absolute path, all previous path components
will be discarded.  An empty last part will result in a path that
ends with a separator.

### aw.extract_agent_spec(agent, name=None)

Extract a normalized AgentSpec from any aw agent.

Introspects the agent’s class, config, tools, and validators
to build a format-agnostic intermediate representation.

* **Parameters:**
  * **agent** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – An aw agent instance (LoadingAgent, PreparationAgent, etc.)
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Override name (defaults to class name)
* **Return type:**
  [`AgentSpec`](aw.translators.html.md#aw.translators.AgentSpec)
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

### aw.functional_validator(try_function, success_check=None)

Create a validator that tries to use the artifact for its purpose.

This validator actually attempts to use the artifact in the way it’s
intended to be used (e.g., try to visualize the data, try to train a model).

* **Parameters:**
  * **try_function** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Function that tries to use the artifact
  * **success_check** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – Optional function to check if result is acceptable
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]
* **Returns:**
  A validator function

### Example

```pycon
>>> def try_cosmo(df):
...     return cosmograph.cosmo(df, points_x_by='x', points_y_by='y')
>>> validate = functional_validator(try_cosmo)
>>> validate(df)
```

### aw.get_numeric_columns(df, exclude_nulls=True)

Get numeric columns from DataFrame.

* **Parameters:**
  * **df** – pandas DataFrame
  * **exclude_nulls** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to exclude columns with null values
* **Returns:**
  Generator of column names

### Example

```pycon
>>> for col in get_numeric_columns(df):
...     print(col)
```

### aw.has_attributes(\*\*required_attrs)

Validator that checks artifact has required attributes.

### Example

```pycon
>>> validate = has_attributes(shape=lambda s: len(s) == 2, columns=lambda c: len(c) > 0)
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

### aw.infer_cosmo_params(df)

Infer suitable cosmograph parameters from DataFrame.

* **Parameters:**
  **df** – pandas DataFrame
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dict of suggested parameters for cosmograph.cosmo()

### Example

```pycon
>>> params = infer_cosmo_params(df)
>>> from cosmograph import cosmo
>>> cosmo(df, **params)
```

### aw.infer_loader_from_extension(extension)

Infer pandas loader function from file extension.

* **Parameters:**
  **extension** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – File extension (e.g., ‘.csv’, ‘.json’)
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Name of pandas loader function

### Example

```pycon
>>> infer_loader_from_extension('.csv')
'read_csv'
>>> infer_loader_from_extension('.xlsx')
'read_excel'
```

### aw.infer_loader_params(extension, sample_text=None)

Infer parameters for pandas loader based on file characteristics.

* **Parameters:**
  * **extension** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – File extension
  * **sample_text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional sample of file content
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dict of parameters to pass to loader

### Example

```pycon
>>> params = infer_loader_params('.csv', 'a,b,c\n1,2,3')
>>> params.get('sep')
','
```

### aw.info_dict_validator(compute_info, check_info)

Create a validator that computes info then checks it.

This is a two-stage validator:

1. Compute information about the artifact (e.g., df.shape, df.dtypes)
2. Check if that information meets requirements

* **Parameters:**
  * **compute_info** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Function to extract info from artifact
  * **check_info** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Function to check if info is acceptable
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]
* **Returns:**
  A validator function

### Example

```pycon
>>> def compute_df_info(df):
...     return {'shape': df.shape, 'null_count': df.isnull().sum().sum()}
>>> def check_df_info(info):
...     if info['null_count'] > 10:
...         return False, "Too many nulls"
...     return True, "OK"
>>> validate = info_dict_validator(compute_df_info, check_df_info)
```

### aw.is_not_empty()

Validator that checks artifact is not empty.

Works for sequences, mappings, DataFrames, etc.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

### aw.is_type(expected_type)

Validator that checks artifact type.

### Example

```pycon
>>> import pandas as pd
>>> validate = is_type(pd.DataFrame)
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

### aw.load_and_prepare(source_uri, target='generic', validator=None, max_retries=3)

Convenience function to load and prepare data in one call.

* **Parameters:**
  * **source_uri** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – URI of data source
  * **target** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Target format/purpose
  * **validator** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Optional custom validator
  * **max_retries** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum retry attempts
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
* **Returns:**
  Tuple of (prepared_dataframe, metadata)

### Example

```pycon
>>> df, meta = load_and_prepare('data.csv', target='cosmo-ready')
```

### aw.load_for_cosmo(source_uri, max_retries=3, strict=False)

Load and prepare data specifically for cosmograph visualization.

* **Parameters:**
  * **source_uri** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – URI of data source
  * **max_retries** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum retry attempts
  * **strict** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, actually calls cosmograph for validation
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
* **Returns:**
  Tuple of (prepared_dataframe, metadata)

### Example

```pycon
>>> df, meta = load_for_cosmo('data.csv')
>>> from cosmograph import cosmo
>>> cosmo(df, **meta['preparing']['metadata']['validation_result']['params'])
```

### aw.openai_assistant_from_spec(spec, , name=None, model='gpt-4')

Render an OpenAI Assistant-style config dict from an [`AgentSpec`](#aw.AgentSpec).

The rendering core of [`to_openai_assistant()`](#aw.to_openai_assistant), driveable by any `AgentSpec`.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### aw.openai_tools_from_spec(spec)

Render OpenAI function-calling tool schemas from an [`AgentSpec`](#aw.AgentSpec).

The rendering core of [`to_openai_tools()`](#aw.to_openai_tools), driveable by any `AgentSpec`.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

### aw.schema_validator(schema)

Create a validator from a schema (Pydantic, JSON Schema, etc.).

* **Parameters:**
  **schema** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A schema object with validation capability
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]
* **Returns:**
  A validator function

### Example

```pycon
>>> from pydantic import BaseModel
>>> class DataSchema(BaseModel):
...     x: float
...     y: float
>>> validate = schema_validator(DataSchema)
>>> validate({'x': 1.0, 'y': 2.0})
(True, {'validated': ...})
```

### aw.strict_cosmo_validator()

Strict validator that actually calls cosmograph.

Requires cosmograph to be installed.

### Example

```pycon
>>> validator = strict_cosmo_validator()
>>> success, info = validator(df)
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)

### aw.to_claude_skill(agent, name=None, description=None, extra_tools=None, disable_model_invocation=False)

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

### aw.to_crewai_yaml(agent, name=None, role=None, goal=None, backstory=None)

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

### aw.to_openai_assistant(agent, name=None, model='gpt-4')

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

### aw.to_openai_tools(agent)

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

### aw.try_cosmo_visualization(df, cosmo_function=None, \*\*cosmo_kwargs)

Try to create a cosmograph visualization.

This is a functional validator that actually attempts the visualization.

* **Parameters:**
  * **df** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – DataFrame to visualize
  * **cosmo_function** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – The cosmograph.cosmo function
  * **\*\*cosmo_kwargs** – Additional arguments for cosmo
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
* **Returns:**
  Tuple of (success, info)

### Example

```pycon
>>> from cosmograph import cosmo
>>> success, info = try_cosmo_visualization(df, cosmo)
```

### aw.workflow_to_crewai_yaml(workflow)

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

### aw.workflow_to_skills(workflow, output_dir)

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

### aw.write_skill_directory(agent, output_dir, name=None, description=None, include_scripts=True)

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

### Modules

| [`base`](aw.base.html.md#module-aw.base)                   | Core protocols and base classes for agentic workflows.                                                                                                                   |
|----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`cosmo`](aw.cosmo.html.md#module-aw.cosmo)                 | Cosmograph-specific validator tool.                                                                                                                                      |
| [`dispatch`](aw.dispatch.html.md#module-aw.dispatch)           | Cheap-first dispatch to a skill: calibrated retrieval first, an LLM adjudicator only on ambiguity.                                                                       |
| [`func_calling`](aw.func_calling.html.md#module-aw.func_calling)   | This module provides robust templates for creating (python) code-generating functions that can translate natural language task descriptions into executable Python code. |
| [`loading`](aw.loading.html.md#module-aw.loading)             | Loading agent for converting data sources to pandas DataFrames.                                                                                                          |
| [`orchestration`](aw.orchestration.html.md#module-aw.orchestration) | Orchestration and workflow management for agentic steps.                                                                                                                 |
| [`preparing`](aw.preparing.html.md#module-aw.preparing)         | Preparation agent for transforming data to meet target requirements.                                                                                                     |
| [`py_gen`](aw.py_gen.html.md#module-aw.py_gen)               | Function makeing and calling                                                                                                                                             |
| [`routing`](aw.routing.html.md#module-aw.routing)             | Configurable routing and decision chains for agentic workflows.                                                                                                          |
| [`tools`](aw.tools.html.md#module-aw.tools)                 | Tools for executing code and other agent actions.                                                                                                                        |
| [`translators`](aw.translators.html.md#module-aw.translators)     | Translators from aw agent specifications to other agent formats.                                                                                                         |
| [`util`](aw.util.html.md#module-aw.util)                   | Utility tools and helper functions for agents.                                                                                                                           |
| [`validation`](aw.validation.html.md#module-aw.validation)       | Validation system supporting three flavors: schema, info-dict, and functional.                                                                                           |
