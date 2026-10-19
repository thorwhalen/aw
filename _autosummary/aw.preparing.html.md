# aw.preparing

Preparation agent for transforming data to meet target requirements.

Implements ReAct pattern with functional validation (try-the-purpose approach).

### Functions

| [`create_preparation_agent`](#aw.preparing.create_preparation_agent)([target, ...])   | Factory function to create a preparation agent.   |
|--------------------------------------------------------------------------------------------|---------------------------------------------------|

### Classes

| [`PreparationAgent`](#aw.preparing.PreparationAgent)([config, target, ...])   | Agent that prepares data to meet target requirements.   |
|--------------------------------------------------------------------------------------------|---------------------------------------------------------|

### *class* aw.preparing.PreparationAgent(config=None, target='generic', target_validator=None)

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

### aw.preparing.create_preparation_agent(target='generic', validator=None, llm=None, max_retries=3)

Factory function to create a preparation agent.

* **Parameters:**
  * **target** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Target format/purpose
  * **validator** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Custom validator function
  * **llm** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – LLM model name or callable
  * **max_retries** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum retry attempts
* **Return type:**
  [`PreparationAgent`](#aw.preparing.PreparationAgent)
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
