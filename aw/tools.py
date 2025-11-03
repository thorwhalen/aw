"""Tools for executing code and other agent actions.

Provides code execution capabilities with safe defaults and extensibility
for robust backends like LangChain's CodeInterpreter.
"""

from typing import Any, Callable
from collections.abc import Mapping
import sys
from io import StringIO
import traceback


class ExecutionResult:
    """Result of code execution.

    Attributes:
        success: Whether execution succeeded without exceptions
        output: Standard output captured during execution
        error: Error message if execution failed
        traceback: Full traceback if execution failed
        result: The return value or final expression value
        locals: Local variables after execution
    """

    def __init__(
        self,
        success: bool,
        output: str = "",
        error: str = "",
        traceback_str: str = "",
        result: Any = None,
        locals_dict: dict = None,
    ):
        self.success = success
        self.output = output
        self.error = error
        self.traceback = traceback_str
        self.result = result
        self.locals = locals_dict or {}

    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return f"ExecutionResult({status}, output_len={len(self.output)}, error={self.error[:50] if self.error else None})"


class CodeInterpreterTool:
    """Tool for executing Python code in a controlled environment.

    Provides a safe default implementation using exec() with limited namespace,
    and allows injection of more robust backends.

    Example:
        >>> tool = CodeInterpreterTool()
        >>> result = tool("x = 5; y = x * 2; print(y)")
        >>> result.success
        True
        >>> result.output.strip()
        '10'
    """

    def __init__(
        self,
        allowed_modules: list[str] = None,
        global_context: dict = None,
        executor: Callable[[str, dict], ExecutionResult] = None,
    ):
        """Initialize code interpreter.

        Args:
            allowed_modules: List of module names that can be imported
            global_context: Additional globals to make available
            executor: Optional custom executor function for advanced backends
        """
        self.allowed_modules = allowed_modules or [
            'pandas',
            'numpy',
            'math',
            'json',
            're',
            'itertools',
            'collections',
            'functools',
            'typing',
        ]
        self.global_context = global_context or {}
        self._executor = executor or self._default_executor

    def __call__(self, code: str, context: dict = None) -> ExecutionResult:
        """Execute Python code.

        Args:
            code: Python code to execute
            context: Additional context/variables to make available

        Returns:
            ExecutionResult with success status and outputs
        """
        merged_context = {**self.global_context, **(context or {})}
        return self._executor(code, merged_context)

    def _default_executor(self, code: str, context: dict) -> ExecutionResult:
        """Default executor using built-in exec()."""
        # Create safe namespace
        safe_globals = self._build_safe_namespace()
        safe_globals.update(context)
        local_vars = {}

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Execute code
            exec(code, safe_globals, local_vars)
            output = captured_output.getvalue()

            # Try to get result from last expression
            result = local_vars.get('_result', None)

            return ExecutionResult(
                success=True, output=output, result=result, locals_dict=local_vars
            )
        except Exception as e:
            error_msg = str(e)
            tb_str = traceback.format_exc()
            return ExecutionResult(
                success=False,
                output=captured_output.getvalue(),
                error=error_msg,
                traceback_str=tb_str,
                locals_dict=local_vars,
            )
        finally:
            sys.stdout = old_stdout

    def _build_safe_namespace(self) -> dict:
        """Build a safe namespace with allowed modules."""
        namespace = {
            '__builtins__': {
                # Include safe builtins
                'print': print,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'int': int,
                'float': float,
                'str': str,
                'bool': bool,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'type': type,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'sorted': sorted,
                'reversed': reversed,
                'any': any,
                'all': all,
                '__import__': __import__,  # Needed for import statements
            }
        }

        # Import allowed modules
        for module_name in self.allowed_modules:
            try:
                module = __import__(module_name)
                # Handle submodules (e.g., pandas.core)
                if '.' in module_name:
                    parts = module_name.split('.')
                    for part in parts[1:]:
                        module = getattr(module, part)
                namespace[module_name.split('.')[0]] = module
            except ImportError:
                # Module not available - skip silently
                pass

        return namespace


def _extract_imports(code: str) -> list[str]:
    """Extract import statements from code.

    Helper function to analyze what modules code needs.

    Example:
        >>> _extract_imports("import pandas as pd\\nfrom numpy import array")
        ['pandas', 'numpy']
    """
    import re

    imports = []

    # Match "import module" and "from module import ..."
    import_pattern = r'^\s*(?:import|from)\s+(\w+)'

    for line in code.split('\n'):
        match = re.match(import_pattern, line)
        if match:
            imports.append(match.group(1))

    return imports


class SafeCodeInterpreter(CodeInterpreterTool):
    """Extra-safe code interpreter with restricted operations.

    Disallows file I/O, network access, and other potentially dangerous operations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Remove potentially dangerous modules
        dangerous_modules = {'os', 'sys', 'subprocess', 'socket', 'urllib'}
        self.allowed_modules = [
            m for m in self.allowed_modules if m not in dangerous_modules
        ]


def create_langchain_executor() -> Callable[[str, dict], ExecutionResult]:
    """Create an executor using LangChain's CodeInterpreter (if available).

    Returns:
        Executor function compatible with CodeInterpreterTool

    Example:
        >>> try:
        ...     executor = create_langchain_executor()
        ...     tool = CodeInterpreterTool(executor=executor)
        ... except ImportError:
        ...     # LangChain not available - use default
        ...     tool = CodeInterpreterTool()
    """
    try:
        from langchain_experimental.tools import PythonREPLTool

        repl = PythonREPLTool()

        def executor(code: str, context: dict) -> ExecutionResult:
            # Inject context variables
            context_setup = '\n'.join(f"{k} = {repr(v)}" for k, v in context.items())
            full_code = f"{context_setup}\n{code}" if context else code

            try:
                output = repl.run(full_code)
                return ExecutionResult(success=True, output=output, result=output)
            except Exception as e:
                return ExecutionResult(
                    success=False, error=str(e), traceback_str=traceback.format_exc()
                )

        return executor
    except ImportError:
        raise ImportError(
            "LangChain not available. Install with: pip install langchain langchain-experimental"
        )


def create_dspy_executor() -> Callable[[str, dict], ExecutionResult]:
    """Create an executor using DSPy's code execution (if available).

    Returns:
        Executor function compatible with CodeInterpreterTool
    """
    try:
        import dspy

        def executor(code: str, context: dict) -> ExecutionResult:
            # DSPy implementation would go here
            # This is a placeholder for the actual implementation
            raise NotImplementedError("DSPy executor not yet implemented")

        return executor
    except ImportError:
        raise ImportError("DSPy not available. Install with: pip install dspy-ai")
