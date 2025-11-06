"""Loading agent for converting data sources to pandas DataFrames.

Implements the ReAct pattern to iteratively try different loading strategies
until successful or max retries reached.
"""

from typing import Any
from collections.abc import MutableMapping

from aw.base import StepConfig, Context
from aw.tools import CodeInterpreterTool, ExecutionResult
from aw.util import (
    FileSamplerTool,
    infer_loader_from_extension,
    infer_loader_params,
    compute_dataframe_info,
    default_llm_factory,
)
from aw.validation import is_type, is_not_empty, all_validators


class LoadingAgent:
    """Agent that loads data from various sources into pandas DataFrames.

    Uses ReAct loop:
    1. Thought: Analyze source (extension, sample) to choose loader
    2. Action: Generate code to load data
    3. Observe: Execute code and capture result/error
    4. Validate: Check if result is valid DataFrame
    5. Repeat or finish

    Example:
        >>> agent = LoadingAgent()
        >>> context = Context()
        >>> df, metadata = agent.execute('/path/to/data.csv', context)
    """

    def __init__(self, config: StepConfig = None):
        """Initialize loading agent.

        Args:
            config: Configuration for the agent (LLM, validator, tools, etc.)
        """
        self.config = config or StepConfig()
        self.llm = self.config.resolve_llm()

        # Set up tools
        self.file_sampler = FileSamplerTool()
        self.code_interpreter = CodeInterpreterTool()

        # Set up validator (DataFrame type + not empty)
        if self.config.validator is None:
            import pandas as pd

            self.validator = all_validators(is_type(pd.DataFrame), is_not_empty())
        else:
            self.validator = self.config.resolve_validator()

    def execute(
        self, source_uri: str, context: MutableMapping[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        """Execute loading agent to load data from source.

        Args:
            source_uri: URI of data source (file path or URL)
            context: Context for storing intermediate results

        Returns:
            Tuple of (dataframe, metadata)
        """
        # Initialize attempt history
        attempts = []

        # Step 1: Sample the file
        file_info = self.file_sampler(source_uri)
        attempts.append({'step': 'file_sample', 'result': file_info})

        if 'error' in file_info:
            return None, {
                'success': False,
                'error': file_info['error'],
                'attempts': attempts,
            }

        # Try to load with increasingly sophisticated approaches
        for attempt_num in range(self.config.max_retries):
            # Step 2: Generate loading code
            code = self._generate_loading_code(file_info, attempts, attempt_num)

            attempts.append(
                {'attempt': attempt_num, 'step': 'generate_code', 'code': code}
            )

            # Step 3: Execute code
            exec_result = self.code_interpreter(code)
            attempts.append(
                {
                    'attempt': attempt_num,
                    'step': 'execute',
                    'success': exec_result.success,
                    'output': exec_result.output,
                    'error': exec_result.error,
                }
            )

            if not exec_result.success:
                # Execution failed - try again with error info
                continue

            # Get the DataFrame from execution result
            df = exec_result.locals.get('df')

            # Step 4: Validate
            is_valid, validation_info = self.validator(df)
            attempts.append(
                {
                    'attempt': attempt_num,
                    'step': 'validate',
                    'success': is_valid,
                    'info': validation_info,
                }
            )

            if is_valid:
                # Success! Compute metadata and return
                df_info = compute_dataframe_info(df)
                metadata = {
                    'success': True,
                    'source_uri': source_uri,
                    'file_info': file_info,
                    'df_info': df_info,
                    'attempts': attempts,
                    'num_attempts': attempt_num + 1,
                }

                # Store in context
                context['loading'] = {'df': df, 'metadata': metadata}

                return df, metadata

        # All attempts failed
        return None, {
            'success': False,
            'error': 'Max retries exceeded',
            'attempts': attempts,
        }

    def _generate_loading_code(
        self, file_info: dict, attempts: list[dict], attempt_num: int
    ) -> str:
        """Generate Python code to load the data.

        Uses LLM for sophisticated reasoning, or falls back to heuristics.

        Args:
            file_info: Information about the file
            attempts: History of previous attempts
            attempt_num: Current attempt number

        Returns:
            Python code string to execute
        """
        # For first attempt, use simple heuristics
        if attempt_num == 0:
            return self._generate_initial_code(file_info)

        # For subsequent attempts, use LLM or refined heuristics
        return self._generate_retry_code(file_info, attempts)

    def _generate_initial_code(self, file_info: dict) -> str:
        """Generate initial loading code based on file info."""
        extension = file_info.get('extension', '.csv')
        uri = file_info.get('uri')

        # Infer loader and params
        loader_func = infer_loader_from_extension(extension)
        sample_text = file_info.get('sample_text')
        params = infer_loader_params(extension, sample_text)

        # Build params string
        params_str = ', '.join(f"{k}={repr(v)}" for k, v in params.items())
        if params_str:
            params_str = ', ' + params_str

        code = f"""import pandas as pd
df = pd.{loader_func}({repr(uri)}{params_str})
"""
        return code

    def _generate_retry_code(self, file_info: dict, attempts: list[dict]) -> str:
        """Generate retry code based on previous failures.

        Analyzes error messages to adjust parameters.
        """
        # Get last error
        last_exec = [a for a in attempts if a.get('step') == 'execute'][-1]
        error_msg = last_exec.get('error', '')

        extension = file_info.get('extension', '.csv')
        uri = file_info.get('uri')
        loader_func = infer_loader_from_extension(extension)

        # Adjust parameters based on error
        params = {}

        if 'UnicodeDecodeError' in error_msg:
            params['encoding'] = 'latin-1'
        elif 'ParserError' in error_msg or 'delimiter' in error_msg.lower():
            # Try different delimiter
            params['sep'] = '\\t' if 'sep=,' in str(attempts) else ';'
        elif 'FileNotFoundError' in error_msg:
            # Try with different path interpretation
            params = {}

        # Try with error_bad_lines parameter for problematic CSVs
        if loader_func == 'read_csv' and 'ParserError' in error_msg:
            params['on_bad_lines'] = 'skip'

        params_str = ', '.join(f"{k}={repr(v)}" for k, v in params.items())
        if params_str:
            params_str = ', ' + params_str

        code = f"""import pandas as pd
df = pd.{loader_func}({repr(uri)}{params_str})
"""
        return code


def create_loading_agent(
    llm: str = None, validator: Any = None, max_retries: int = 3
) -> LoadingAgent:
    """Factory function to create a loading agent.

    Args:
        llm: LLM model name or callable
        validator: Custom validator (uses default if None)
        max_retries: Maximum retry attempts

    Returns:
        Configured LoadingAgent

    Example:
        >>> agent = create_loading_agent(llm="gpt-4", max_retries=5)
    """
    config = StepConfig(
        llm=llm or "gpt-4", validator=validator, max_retries=max_retries
    )
    return LoadingAgent(config)
