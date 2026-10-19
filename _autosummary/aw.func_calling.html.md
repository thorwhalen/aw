# aw.func_calling

This module provides robust templates for creating (python) code-generating functions
that can translate natural language task descriptions into executable Python code.

### Functions

| [`extract_function_from_code`](#aw.func_calling.extract_function_from_code)(code_str_for_func, \*)   | Extract a function from a string of Python code.                  |
|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [`make_code_generator`](#aw.func_calling.make_code_generator)([template, code_schema, ...])   | Create a code generation function from a template and schema.     |
| [`task_to_function`](#aw.func_calling.task_to_function)(task[, output_schema, ...])        | End-to-end: Convert a task description to an executable function. |

### aw.func_calling.extract_function_from_code(code_str_for_func, , make_picklable=False, strict=True, verbose=True)

Extract a function from a string of Python code.

* **Parameters:**
  * **code_str_for_func** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – String containing a Python function definition
  * **make_picklable** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, make the function picklable by adding it to \_\_main_\_
  * **strict** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, enforce strict validation (no multiple functions, no syntax errors)
  * **verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, print status messages
* **Return type:**
  `callable`
* **Returns:**
  The function object
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If no function is found or multiple functions exist

### Example

```pycon
>>> code = '''
... def add(a: int, b: int) -> dict:
...     return {"sum": a + b}
... '''
>>> func = extract_function_from_code(code, verbose=False)
>>> func(2, 3)
{'sum': 5}
```

### aw.func_calling.make_code_generator(template='\\nYou are a Python code generator. Your task is to create a complete, executable Python function.\\n\\n## TASK DESCRIPTION\\n{task}\\n\\n## OUTPUT SCHEMA\\nThe function should return a value that conforms to this JSON schema:\\n\`\`\`json\\n{output_schema}\\n\`\`\`\\n\\nIf output_schema is null or empty, the function can return any appropriate value.\\n\\n## REQUIREMENTS\\n\\n### Function Signature\\n- Function name: \`{name}\`\\n- Parameters: Extract from the task description (indicated by {{braces}})\\n- All parameters should have type hints where possible\\n- Use clear, descriptive parameter names\\n\\n### Code Quality\\n- Write ONLY the function definition (no imports in function body, no examples, no explanations)\\n- Include a minimal docstring (one line describing what it does)\\n- Use type hints for parameters and return value\\n- Handle edge cases appropriately (e.g., empty inputs, None values)\\n- Use built-in functions and standard library where possible\\n- Keep the code simple and readable\\n\\n### Output Format\\n- If returning a dict matching a schema, ensure all required fields are present\\n- If the schema specifies types, ensure they match (e.g., "number" → int or float)\\n- Return values directly without unnecessary wrapping\\n\\n### Code Style\\n- Follow PEP 8 conventions\\n- Use meaningful variable names\\n- Prefer comprehensions over loops where readable\\n- No print statements or side effects unless the task requires them\\n\\n## CRITICAL INSTRUCTIONS\\n- Respond with ONLY the function definition\\n- Do NOT include: imports, examples, explanations, markdown formatting, or test code\\n- Do NOT wrap the code in markdown code blocks\\n- The response must be valid Python that can be directly executed with \`exec()\`\\n\\n## EXAMPLE FORMAT (for reference only, do not include this in your response)\\n\`\`\`python\\ndef example_function(param1: int, param2: str) -> dict:\\n    """Brief description of what this does."""\\n    # function body\\n    return {{"result": param1}}\\n\`\`\`\\n\\nNOW GENERATE THE FUNCTION.\\n', code_schema={'name': 'generated_python_function', 'schema': {'properties': {'code': {'description': 'Complete Python function definition as a string', 'type': 'string'}}, 'required': ['code'], 'type': 'object'}}, prompt_json_function_maker=None)

Create a code generation function from a template and schema.

* **Parameters:**
  * **template** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The prompt template for code generation
  * **code_schema** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – JSON schema defining the output format
  * **prompt_json_function_maker** – Factory function (defaults to oa.prompt_json_function)
* **Returns:**
  A function that generates code from task descriptions

#### NOTE
The returned function requires all template variables to be provided.
See CODE_WRITING_TEMPLATE for required variables (task, output_schema, name).

### aw.func_calling.task_to_function(task, output_schema=None, name='generated_function', code_generator=None, \*\*generator_kwargs)

End-to-end: Convert a task description to an executable function.

* **Parameters:**
  * **task** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Natural language description of the function’s purpose
  * **output_schema** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – JSON schema for the return value (optional)
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name for the generated function
  * **code_generator** – Code generation function (creates one if None)
  * **\*\*generator_kwargs** – Additional arguments for code generator
* **Return type:**
  `callable`
* **Returns:**
  Executable Python function

#### NOTE
This function requires an LLM backend (via code_generator or the ‘oa’ package).
Cannot be tested in doctests without mocking the LLM calls.
