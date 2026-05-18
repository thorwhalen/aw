"""
This module provides robust templates for creating (python) code-generating functions
that can translate natural language task descriptions into executable Python code.
"""

import json
import inspect
from typing import Any


# ============================================================================
# Code Writing Template
# ============================================================================

CODE_WRITING_TEMPLATE = """
You are a Python code generator. Your task is to create a complete, executable Python function.

## TASK DESCRIPTION
{task}

## OUTPUT SCHEMA
The function should return a value that conforms to this JSON schema:
```json
{output_schema}
```

If output_schema is null or empty, the function can return any appropriate value.

## REQUIREMENTS

### Function Signature
- Function name: `{name}`
- Parameters: Extract from the task description (indicated by {{braces}})
- All parameters should have type hints where possible
- Use clear, descriptive parameter names

### Code Quality
- Write ONLY the function definition (no imports in function body, no examples, no explanations)
- Include a minimal docstring (one line describing what it does)
- Use type hints for parameters and return value
- Handle edge cases appropriately (e.g., empty inputs, None values)
- Use built-in functions and standard library where possible
- Keep the code simple and readable

### Output Format
- If returning a dict matching a schema, ensure all required fields are present
- If the schema specifies types, ensure they match (e.g., "number" → int or float)
- Return values directly without unnecessary wrapping

### Code Style
- Follow PEP 8 conventions
- Use meaningful variable names
- Prefer comprehensions over loops where readable
- No print statements or side effects unless the task requires them

## CRITICAL INSTRUCTIONS
- Respond with ONLY the function definition
- Do NOT include: imports, examples, explanations, markdown formatting, or test code
- Do NOT wrap the code in markdown code blocks
- The response must be valid Python that can be directly executed with `exec()`

## EXAMPLE FORMAT (for reference only, do not include this in your response)
```python
def example_function(param1: int, param2: str) -> dict:
    \"\"\"Brief description of what this does.\"\"\"
    # function body
    return {{"result": param1}}
```

NOW GENERATE THE FUNCTION.
"""


# ============================================================================
# ALTERNATIVE: More Structured Template with Better Guidance
# ============================================================================

STRUCTURED_CODE_WRITING_TEMPLATE = """
# Python Function Generator

You will generate a single Python function based on specifications below.

## 1. TASK
{task}

## 2. SPECIFICATIONS
- **Function Name**: `{name}`
- **Input Parameters**: Identified by `{{variable}}` patterns in task description
- **Output Format**: Must conform to this JSON schema:
```json
{output_schema}
```
(If schema is null/empty, return type can be determined from task context)

## 3. IMPLEMENTATION GUIDE

### Parameter Extraction
Scan the task description for {{variable_name}} patterns. These become function parameters.
Example: "Add {{a}} and {{b}}" → parameters are `a` and `b`

### Type Hints
- Add type hints to all parameters (int, str, float, list, dict, Any, etc.)
- Add return type hint based on output schema:
  * Schema "type": "object" → return type `dict`
  * Schema "type": "array" → return type `list`
  * Schema "type": "string" → return type `str`
  * Schema "type": "number" or "integer" → return type `int` or `float`
  * No schema → use `Any` or infer from task

### Output Schema Conformance
When a JSON schema is provided:
1. Return value must match the schema structure exactly
2. Include all "required" fields from schema
3. Use correct Python types for each field:
   - JSON "string" → Python str
   - JSON "number"/"integer" → Python int/float
   - JSON "boolean" → Python bool
   - JSON "array" → Python list
   - JSON "object" → Python dict
4. For nested structures, preserve the hierarchy

### Code Quality Standards
✓ Single function definition only (no external helpers or imports)
✓ One-line docstring summarizing purpose
✓ Type hints on parameters and return value
✓ Handle edge cases (None, empty collections, type mismatches)
✓ Return value matching schema requirements
✓ No debugging print(), comments outside function, or example calls
✓ Valid Python syntax (must pass compile())
✓ PEP 8 compliant naming and style

## 4. CRITICAL OUTPUT REQUIREMENTS
Return ONLY the function definition as plain Python code.

DO NOT INCLUDE:
- Markdown code fences (```) or formatting
- Import statements (unless absolutely essential)
- Example usage or test code
- Explanatory comments or documentation outside the function
- Multiple function definitions

The output must be executable with `exec()` and should define exactly one function.

---
GENERATE THE FUNCTION:
"""


# ============================================================================
# MINIMAL: Concise Template for Simple Cases
# ============================================================================

MINIMAL_CODE_WRITING_TEMPLATE = """
Write a Python function that does: {task}

Requirements:
- Function name: {name}
- Parameters: Extract from {{variable}} patterns
- Return format: {output_schema}
- Include type hints and docstring
- Return only the function definition (no examples, imports, or markdown)

Generate the function:
"""


# ============================================================================
# Schema Definitions for Code Output
# ============================================================================

# Simple schema: just the code string
CODE_DEFINITION_SCHEMA = {
    "name": "generated_python_function",
    "schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Complete Python function definition as a string",
            },
        },
        "required": ["code"],
    },
}

# Detailed schema: includes metadata
DETAILED_CODE_DEFINITION_SCHEMA = {
    "name": "detailed_python_function",
    "schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Complete Python function definition",
            },
            "function_name": {"type": "string", "description": "Name of the function"},
            "parameters": {
                "type": "array",
                "description": "List of parameter names",
                "items": {"type": "string"},
            },
            "docstring": {"type": "string", "description": "The function docstring"},
        },
        "required": ["code", "function_name"],
    },
}


# ============================================================================
# Helper Functions
# ============================================================================


def extract_function_from_code(
    code_str_for_func: str,
    *,
    make_picklable: bool = False,
    strict: bool = True,  # NEW: toggle strict validation
    verbose: bool = True,  # NEW: toggle print statements
) -> callable:
    """
    Extract a function from a string of Python code.

    Args:
        code_str_for_func: String containing a Python function definition
        make_picklable: If True, make the function picklable by adding it to __main__
        strict: If True, enforce strict validation (no multiple functions, no syntax errors)
        verbose: If True, print status messages

    Returns:
        The function object

    Raises:
        ValueError: If no function is found or multiple functions exist

    Example:

        >>> code = '''
        ... def add(a: int, b: int) -> dict:
        ...     return {"sum": a + b}
        ... '''
        >>> func = extract_function_from_code(code, verbose=False)
        >>> func(2, 3)
        {'sum': 5}

    """

    if strict:
        # Pre-compile check
        try:
            compile(code_str_for_func, "<generated>", "exec")
        except SyntaxError as e:
            raise ValueError(f"Generated code has syntax error: {e}")

    namespace = {}
    exec(code_str_for_func, globals(), namespace)

    functions = [obj for obj in namespace.values() if inspect.isfunction(obj)]

    if not functions:
        raise ValueError("No function found in code string")

    if strict and len(functions) > 1:
        raise ValueError(f"Multiple functions: {[f.__name__ for f in functions]}")

    func = functions[0]

    if make_picklable:
        name = func.__name__
        if name in sys.modules["__main__"].__dict__:
            if verbose and inspect.isfunction(sys.modules["__main__"].__dict__[name]):
                print(f"WARNING: Overwriting '{name}' in __main__")

        sys.modules["__main__"].__dict__[name] = func
        if verbose:
            print(f"Loaded function: {name} (picklable)")
    elif verbose:
        print(f"Loaded function: {func.__name__} (not picklable)")

    return func


def make_code_generator(
    template: str = CODE_WRITING_TEMPLATE,
    code_schema: dict = CODE_DEFINITION_SCHEMA,
    prompt_json_function_maker=None,
):
    """
    Create a code generation function from a template and schema.

    Args:
        template: The prompt template for code generation
        code_schema: JSON schema defining the output format
        prompt_json_function_maker: Factory function (defaults to oa.prompt_json_function)

    Returns:
        A function that generates code from task descriptions

    Note:
        The returned function requires all template variables to be provided.
        See CODE_WRITING_TEMPLATE for required variables (task, output_schema, name).
    """
    if prompt_json_function_maker is None:
        # Try to import from oa
        try:
            from oa import prompt_json_function

            prompt_json_function_maker = prompt_json_function
        except ImportError:
            raise ValueError(
                "Must provide prompt_json_function_maker or have 'oa' package installed"
            )

    return prompt_json_function_maker(template, json_schema=code_schema)


def task_to_function(
    task: str,
    output_schema: dict | str | None = None,
    name: str = "generated_function",
    code_generator=None,
    **generator_kwargs,
) -> callable:
    """
    End-to-end: Convert a task description to an executable function.

    Args:
        task: Natural language description of the function's purpose
        output_schema: JSON schema for the return value (optional)
        name: Name for the generated function
        code_generator: Code generation function (creates one if None)
        **generator_kwargs: Additional arguments for code generator

    Returns:
        Executable Python function

    Note:
        This function requires an LLM backend (via code_generator or the 'oa' package).
        Cannot be tested in doctests without mocking the LLM calls.
    """
    if code_generator is None:
        code_generator = make_code_generator()

    # Format the output schema as JSON string if it's a dict
    if isinstance(output_schema, dict):
        output_schema_str = json.dumps(output_schema, indent=2)
    else:
        output_schema_str = output_schema or "null"

    # Generate code
    result = code_generator(
        task=task, output_schema=output_schema_str, name=name, **generator_kwargs
    )

    # Extract function from generated code
    code_str = result["code"]
    return extract_function_from_code(code_str)


# ============================================================================
# Demonstration
# ============================================================================


def _demo():
    """Show example of how to use the templates."""
    print("=" * 70)
    print("RECOMMENDED TEMPLATE:")
    print("=" * 70)
    print(CODE_WRITING_TEMPLATE)
    print("\n" + "=" * 70)
    print("CODE DEFINITION SCHEMA:")
    print("=" * 70)
    print(json.dumps(CODE_DEFINITION_SCHEMA, indent=2))

    print("\n" + "=" * 70)
    print("USAGE EXAMPLE:")
    print("=" * 70)
    print(
        """
from oa import prompt_json_function
from code_generation_improved import CODE_WRITING_TEMPLATE, CODE_DEFINITION_SCHEMA

# Create the code generator
write_code = prompt_json_function(
    CODE_WRITING_TEMPLATE,
    json_schema=CODE_DEFINITION_SCHEMA
)

# Generate code for a task
task = 'Add numbers {a} and {b}'
output_schema = {"type": "object", "properties": {"sum": {"type": "number"}}}
result = write_code(
    task=task,
    output_schema=json.dumps(output_schema),
    name='add_numbers'
)

# Execute the generated code
func = extract_function_from_code(result['code'])
print(func(2, 3))  # {'sum': 5}
    """
    )


if __name__ == "__main__":
    _demo()
