# aw.tools

Tools for executing code and other agent actions.

Provides code execution capabilities with safe defaults and extensibility
for robust backends like LangChain’s CodeInterpreter.

### Functions

| [`create_dspy_executor`](#aw.tools.create_dspy_executor)()      | Create an executor using DSPy's code execution (if available).       |
|------------------------------------------------------------------------------|----------------------------------------------------------------------|
| [`create_langchain_executor`](#aw.tools.create_langchain_executor)() | Create an executor using LangChain's CodeInterpreter (if available). |

### Classes

| [`CodeInterpreterTool`](#aw.tools.CodeInterpreterTool)([allowed_modules, ...])    | Tool for executing Python code in a controlled environment.   |
|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| [`ExecutionResult`](#aw.tools.ExecutionResult)(success[, output, error, ...]) | Result of code execution.                                     |
| [`SafeCodeInterpreter`](#aw.tools.SafeCodeInterpreter)(\*\*kwargs)                | Extra-safe code interpreter with restricted operations.       |

### *class* aw.tools.CodeInterpreterTool(allowed_modules=None, global_context=None, executor=None)

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

### *class* aw.tools.ExecutionResult(success, output='', error='', traceback_str='', result=None, locals_dict=None)

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

### *class* aw.tools.SafeCodeInterpreter(\*\*kwargs)

Bases: [`CodeInterpreterTool`](#aw.tools.CodeInterpreterTool)

Extra-safe code interpreter with restricted operations.

Disallows file I/O, network access, and other potentially dangerous operations.

### aw.tools.create_dspy_executor()

Create an executor using DSPy’s code execution (if available).

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)], [`ExecutionResult`](#aw.tools.ExecutionResult)]
* **Returns:**
  Executor function compatible with CodeInterpreterTool

### aw.tools.create_langchain_executor()

Create an executor using LangChain’s CodeInterpreter (if available).

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)], [`ExecutionResult`](#aw.tools.ExecutionResult)]
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
