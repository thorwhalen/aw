# aw.orchestration

Orchestration and workflow management for agentic steps.

Provides utilities to chain multiple agentic steps together and manage
the overall workflow.

### Functions

| [`create_cosmo_prep_workflow`](#aw.orchestration.create_cosmo_prep_workflow)([...])                 | Create a workflow specifically for cosmograph preparation.       |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| [`create_data_prep_workflow`](#aw.orchestration.create_data_prep_workflow)([loading_config, ...])  | Factory to create a data preparation workflow.                   |
| [`load_and_prepare`](#aw.orchestration.load_and_prepare)(source_uri[, target, ...])       | Convenience function to load and prepare data in one call.       |
| [`load_for_cosmo`](#aw.orchestration.load_for_cosmo)(source_uri[, max_retries, strict]) | Load and prepare data specifically for cosmograph visualization. |

### Classes

| [`AgenticWorkflow`](#aw.orchestration.AgenticWorkflow)([context])     | Orchestrates a chain of agentic steps.        |
|---------------------------------------------------------------------------------|-----------------------------------------------|
| [`InteractiveWorkflow`](#aw.orchestration.InteractiveWorkflow)([context]) | Workflow with human-in-the-loop capabilities. |

### *class* aw.orchestration.AgenticWorkflow(context=None)

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
  [`AgenticWorkflow`](#aw.orchestration.AgenticWorkflow)
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

### *class* aw.orchestration.InteractiveWorkflow(context=None)

Bases: [`AgenticWorkflow`](#aw.orchestration.AgenticWorkflow)

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
  [`InteractiveWorkflow`](#aw.orchestration.InteractiveWorkflow)

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

### aw.orchestration.create_cosmo_prep_workflow(cosmo_validator=None, max_retries=3)

Create a workflow specifically for cosmograph preparation.

* **Parameters:**
  * **cosmo_validator** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Custom cosmo validator (uses default if None)
  * **max_retries** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum retries for each step
* **Return type:**
  [`AgenticWorkflow`](#aw.orchestration.AgenticWorkflow)
* **Returns:**
  Configured workflow

### Example

```pycon
>>> workflow = create_cosmo_prep_workflow()
>>> prepared_df, metadata = workflow.run('data.csv')
```

### aw.orchestration.create_data_prep_workflow(loading_config=None, preparing_config=None, target='generic')

Factory to create a data preparation workflow.

Creates a workflow with LoadingAgent -> PreparationAgent.

* **Parameters:**
  * **loading_config** ([`StepConfig`](aw.base.html.md#aw.base.StepConfig)) – Configuration for loading agent
  * **preparing_config** ([`StepConfig`](aw.base.html.md#aw.base.StepConfig)) – Configuration for preparing agent
  * **target** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Target format for preparation
* **Return type:**
  [`AgenticWorkflow`](#aw.orchestration.AgenticWorkflow)
* **Returns:**
  Configured workflow

### Example

```pycon
>>> workflow = create_data_prep_workflow(target='cosmo-ready')
>>> df, metadata = workflow.run('/path/to/data.csv')
```

### aw.orchestration.load_and_prepare(source_uri, target='generic', validator=None, max_retries=3)

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

### aw.orchestration.load_for_cosmo(source_uri, max_retries=3, strict=False)

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
