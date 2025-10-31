"""Preparation agent for transforming data to meet target requirements.

Implements ReAct pattern with functional validation (try-the-purpose approach).
"""

from typing import Any, Callable
from collections.abc import MutableMapping

from aw.base import StepConfig, Context
from aw.tools import CodeInterpreterTool
from aw.utils import compute_dataframe_info, default_llm_factory
from aw.validation import functional_validator


class PreparationAgent:
    """Agent that prepares data to meet target requirements.

    Uses ReAct loop with functional validation:
    1. Thought: Analyze current data state vs. requirements
    2. Action: Generate transformation code
    3. Observe: Execute code and capture result
    4. Validate: Try to use data for its purpose (e.g., visualization)
    5. Repeat or finish

    Example:
        >>> agent = PreparationAgent(target='cosmo-ready')
        >>> context = Context({'loading': {'df': df}})
        >>> prepared_df, metadata = agent.execute(df, context)
    """

    def __init__(
        self,
        config: StepConfig = None,
        target: str = 'generic',
        target_validator: Callable = None,
    ):
        """Initialize preparation agent.

        Args:
            config: Configuration for the agent
            target: Target format/purpose (e.g., 'cosmo-ready', 'ml-ready')
            target_validator: Custom validator for target requirements
        """
        self.config = config or StepConfig()
        self.llm = self.config.resolve_llm()
        self.target = target

        # Set up code interpreter
        self.code_interpreter = CodeInterpreterTool()

        # Set up validator
        if target_validator:
            self.validator = target_validator
        elif self.config.validator:
            self.validator = self.config.resolve_validator()
        else:
            # Use a basic validator
            self.validator = lambda df: (True, {'note': 'No validation specified'})

    def execute(
        self, input_df: Any, context: MutableMapping[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        """Execute preparation agent to transform data.

        Args:
            input_df: Input DataFrame to prepare
            context: Context for storing intermediate results

        Returns:
            Tuple of (prepared_dataframe, metadata)
        """
        import pandas as pd

        # Initialize
        attempts = []
        current_df = input_df.copy() if hasattr(input_df, 'copy') else input_df

        # Get initial info
        initial_info = compute_dataframe_info(current_df)
        attempts.append({'step': 'initial_analysis', 'info': initial_info})

        # Iterate with transformations
        for attempt_num in range(self.config.max_retries):
            # Step 1: Generate transformation code
            code = self._generate_transformation_code(
                current_df, initial_info, attempts, attempt_num
            )

            attempts.append(
                {'attempt': attempt_num, 'step': 'generate_code', 'code': code}
            )

            # Step 2: Execute transformation
            exec_result = self.code_interpreter(
                code, context={'df': current_df, 'pd': pd}
            )

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
                # Execution failed - try again
                continue

            # Get transformed DataFrame
            transformed_df = exec_result.locals.get('transformed_df', current_df)

            # Step 3: Validate with target validator
            is_valid, validation_info = self.validator(transformed_df)
            attempts.append(
                {
                    'attempt': attempt_num,
                    'step': 'validate',
                    'success': is_valid,
                    'info': validation_info,
                }
            )

            if is_valid:
                # Success!
                final_info = compute_dataframe_info(transformed_df)
                metadata = {
                    'success': True,
                    'target': self.target,
                    'initial_info': initial_info,
                    'final_info': final_info,
                    'validation_result': validation_info,
                    'attempts': attempts,
                    'num_attempts': attempt_num + 1,
                }

                # Store in context
                context['preparing'] = {'df': transformed_df, 'metadata': metadata}

                return transformed_df, metadata

            # Validation failed but code ran - update current_df and try again
            current_df = transformed_df

        # Max retries exceeded
        return current_df, {
            'success': False,
            'error': 'Max retries exceeded',
            'target': self.target,
            'attempts': attempts,
        }

    def _generate_transformation_code(
        self, df: Any, initial_info: dict, attempts: list[dict], attempt_num: int
    ) -> str:
        """Generate transformation code.

        Args:
            df: Current DataFrame
            initial_info: Initial analysis info
            attempts: History of attempts
            attempt_num: Current attempt number

        Returns:
            Python code string
        """
        if attempt_num == 0:
            return self._generate_initial_transformation(df, initial_info)
        else:
            return self._generate_retry_transformation(df, initial_info, attempts)

    def _generate_initial_transformation(self, df: Any, info: dict) -> str:
        """Generate initial transformation based on target."""
        if self.target == 'cosmo-ready':
            return self._generate_cosmo_transformation(df, info)
        else:
            # Generic: just ensure no nulls
            return """
transformed_df = df.copy()
transformed_df = transformed_df.dropna()
"""

    def _generate_cosmo_transformation(self, df: Any, info: dict) -> str:
        """Generate transformation for cosmograph requirements.

        Cosmograph needs:
        - At least 2 numeric columns for x/y coordinates
        - No nulls in those columns
        """
        numeric_cols = info.get('numeric_columns', [])

        if len(numeric_cols) >= 2:
            # We have numeric columns - just clean them
            return f"""
transformed_df = df.copy()
# Ensure numeric columns are clean
numeric_cols = {numeric_cols[:5]}  # Use first few numeric columns
for col in numeric_cols:
    if col in transformed_df.columns:
        transformed_df[col] = pd.to_numeric(transformed_df[col], errors='coerce')
# Drop rows with nulls in numeric columns
transformed_df = transformed_df.dropna(subset=numeric_cols)
"""
        else:
            # Need to create or convert numeric columns
            return """
import numpy as np
transformed_df = df.copy()

# Try to convert object columns to numeric
for col in transformed_df.select_dtypes(include='object').columns:
    transformed_df[col] = pd.to_numeric(transformed_df[col], errors='coerce')

# If still no numeric columns, create indices
numeric_cols = transformed_df.select_dtypes(include='number').columns
if len(numeric_cols) < 2:
    transformed_df['_x'] = np.arange(len(transformed_df))
    transformed_df['_y'] = np.random.randn(len(transformed_df))

# Drop nulls
transformed_df = transformed_df.dropna()
"""

    def _generate_retry_transformation(
        self, df: Any, initial_info: dict, attempts: list[dict]
    ) -> str:
        """Generate retry transformation based on validation errors."""
        # Get last validation error
        last_validation = [a for a in attempts if a.get('step') == 'validate'][-1]

        error_info = last_validation.get('info', {})
        error_msg = error_info.get('error', '')

        # Adjust based on error
        if 'not numeric' in error_msg.lower():
            return """
import numpy as np
transformed_df = df.copy()
# More aggressive numeric conversion
for col in transformed_df.columns:
    try:
        transformed_df[col] = pd.to_numeric(transformed_df[col], errors='coerce')
    except:
        pass
transformed_df = transformed_df.dropna()
"""
        elif 'too few' in error_msg.lower() or 'column' in error_msg.lower():
            return """
import numpy as np
transformed_df = df.copy()
# Ensure we have enough numeric columns
numeric_cols = list(transformed_df.select_dtypes(include='number').columns)
while len(numeric_cols) < 2:
    new_col = f'_generated_{len(numeric_cols)}'
    transformed_df[new_col] = np.random.randn(len(transformed_df))
    numeric_cols.append(new_col)
transformed_df = transformed_df.dropna()
"""
        else:
            # Generic retry - more aggressive cleaning
            return """
transformed_df = df.copy()
# Drop all non-numeric columns
transformed_df = transformed_df.select_dtypes(include='number')
# Fill nulls with mean
transformed_df = transformed_df.fillna(transformed_df.mean())
"""


def create_preparation_agent(
    target: str = 'generic',
    validator: Callable = None,
    llm: str = None,
    max_retries: int = 3,
) -> PreparationAgent:
    """Factory function to create a preparation agent.

    Args:
        target: Target format/purpose
        validator: Custom validator function
        llm: LLM model name or callable
        max_retries: Maximum retry attempts

    Returns:
        Configured PreparationAgent

    Example:
        >>> agent = create_preparation_agent(
        ...     target='cosmo-ready',
        ...     validator=cosmo_validator,
        ...     max_retries=5
        ... )
    """
    config = StepConfig(
        llm=llm or "gpt-4", validator=validator, max_retries=max_retries
    )
    return PreparationAgent(config=config, target=target, target_validator=validator)
