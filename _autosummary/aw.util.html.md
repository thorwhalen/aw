# aw.util

Utility tools and helper functions for agents.

Includes file sampling, LLM facades, and other common utilities.

### Functions

| [`claude_desktop_config`](#aw.util.claude_desktop_config)([key_path, config_dir])   | Get a FileStore interface to Claude Desktop's configuration.      |
|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [`compute_dataframe_info`](#aw.util.compute_dataframe_info)(df)                      | Compute comprehensive info about a DataFrame.                     |
| [`create_oa_chat`](#aw.util.create_oa_chat)()                                | Create a chat function using the oa package.                      |
| [`create_openai_chat`](#aw.util.create_openai_chat)([model])                     | Create a text-to-text chat function using OpenAI API.             |
| [`default_llm_factory`](#aw.util.default_llm_factory)([model])                    | Factory to create LLM function from various specifications.       |
| [`get_numeric_columns`](#aw.util.get_numeric_columns)(df[, exclude_nulls])        | Get numeric columns from DataFrame.                               |
| [`infer_loader_from_extension`](#aw.util.infer_loader_from_extension)(extension)          | Infer pandas loader function from file extension.                 |
| [`infer_loader_params`](#aw.util.infer_loader_params)(extension[, sample_text])   | Infer parameters for pandas loader based on file characteristics. |

### Classes

| [`FileSamplerTool`](#aw.util.FileSamplerTool)([sample_size])   | Tool to sample and analyze file metadata and content.   |
|-----------------------------------------------------------------------------------|---------------------------------------------------------|

### *class* aw.util.FileSamplerTool(sample_size=1024)

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

### aw.util.claude_desktop_config(key_path='mcpServers', , config_dir=None)

Get a FileStore interface to Claude Desktop’s configuration.

By default, operates on the ‘mcpServers’ section.

Automatically creates the config file if it doesn’t exist, and creates the
mcpServers section if it’s missing.

* **Parameters:**
  * **key_path** – Path to nested section (default: ‘mcpServers’)
  * **config_dir** – Override default config directory location
* **Returns:**
  FileStore interface to Claude Desktop config

### Example

```pycon
>>> # Add/manage MCP servers
>>> mcp = claude_desktop_config()
>>>
>>> # Add a new server
>>> mcp['download'] = {
...     'command': 'python',
...     'args': ['/path/to/agent.py']
... }
>>>
>>> # List configured servers
>>> list(mcp)
>>>
>>> # Remove a server
>>> del mcp['old_server']
```

### aw.util.compute_dataframe_info(df)

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

### aw.util.create_oa_chat()

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

### aw.util.create_openai_chat(model='gpt-4', \*\*default_kwargs)

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

### aw.util.default_llm_factory(model='gpt-4')

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

### aw.util.get_numeric_columns(df, exclude_nulls=True)

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

### aw.util.infer_loader_from_extension(extension)

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

### aw.util.infer_loader_params(extension, sample_text=None)

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
