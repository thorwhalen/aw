# aw.loading

Loading agent for converting data sources to pandas DataFrames.

Implements the ReAct pattern to iteratively try different loading strategies
until successful or max retries reached.

### Functions

| [`create_loading_agent`](#aw.loading.create_loading_agent)([llm, validator, ...])   | Factory function to create a loading agent.   |
|------------------------------------------------------------------------------------------------|-----------------------------------------------|

### Classes

| [`LoadingAgent`](#aw.loading.LoadingAgent)([config])   | Agent that loads data from various sources into pandas DataFrames.   |
|---------------------------------------------------------------------------|----------------------------------------------------------------------|

### *class* aw.loading.LoadingAgent(config=None)

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

### aw.loading.create_loading_agent(llm=None, validator=None, max_retries=3)

Factory function to create a loading agent.

* **Parameters:**
  * **llm** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – LLM model name or callable
  * **validator** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Custom validator (uses default if None)
  * **max_retries** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum retry attempts
* **Return type:**
  [`LoadingAgent`](#aw.loading.LoadingAgent)
* **Returns:**
  Configured LoadingAgent

### Example

```pycon
>>> agent = create_loading_agent(llm="gpt-4", max_retries=5)
```
