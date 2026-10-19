# aw.base

Core protocols and base classes for agentic workflows.

This module provides the foundational abstractions for building AI agents
that follow the AgenticStep protocol with ReAct pattern (Reason-Act-Observe).

### Classes

| [`AgenticStep`](#aw.base.AgenticStep)(\*args, \*\*kwargs)                 | Protocol defining the interface for any agentic step in a workflow.   |
|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| [`Context`](#aw.base.Context)([initial_data])                         | Context for sharing state and artifacts between agentic steps.        |
| [`GlobalConfig`](#aw.base.GlobalConfig)([llm, max_retries, human_in_loop]) | Global configuration with cascading defaults.                         |
| [`StepConfig`](#aw.base.StepConfig)([llm, validator, tools, ...])        | Configuration for an AgenticStep.                                     |
| [`Tool`](#aw.base.Tool)(\*args, \*\*kwargs)                        | Protocol for tools that agents can use.                               |
| [`Validator`](#aw.base.Validator)(\*args, \*\*kwargs)                   | Protocol for validation functions.                                    |

### *class* aw.base.AgenticStep(\*args, \*\*kwargs)

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

### *class* aw.base.Context(initial_data=None)

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

### *class* aw.base.GlobalConfig(llm='gpt-4', max_retries=3, human_in_loop=False)

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
  [`StepConfig`](#aw.base.StepConfig)
* **Returns:**
  A new StepConfig with merged configuration

### *class* aw.base.StepConfig(llm='gpt-4', validator=None, tools=list(), max_retries=3, human_in_loop=False)

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
  [`Validator`](#aw.base.Validator)
* **Returns:**
  A callable that validates artifacts

### *class* aw.base.Tool(\*args, \*\*kwargs)

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

### *class* aw.base.Validator(\*args, \*\*kwargs)

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
