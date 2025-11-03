"""Cosmograph-specific validator tool.

Provides functional validation for cosmograph visualization requirements.
"""

from typing import Any, Callable
from aw.validation import functional_validator


def create_cosmo_validator(
    cosmo_function: Callable = None,
    required_columns: int = 2,
    allow_generated_params: bool = True,
) -> Callable[[Any], tuple[bool, dict]]:
    """Create a validator for cosmograph requirements.

    This validator attempts to actually call cosmograph.cosmo() with the
    DataFrame to see if it's ready for visualization.

    Args:
        cosmo_function: The cosmograph.cosmo function (imported if None)
        required_columns: Minimum number of numeric columns needed
        allow_generated_params: Whether to auto-infer x/y columns

    Returns:
        Validator function following (artifact) -> (success, info) protocol

    Example:
        >>> validator = create_cosmo_validator()  # doctest: +SKIP
        >>> success, info = validator(df)  # doctest: +SKIP
        >>> if success:  # doctest: +SKIP
        ...     print(f"Can visualize with: {info['params']}")  # doctest: +SKIP
    """

    def validate_cosmo_ready(df) -> tuple[bool, dict]:
        """Validate if DataFrame is ready for cosmograph visualization."""
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            return False, {'error': 'Not a DataFrame', 'type': type(df).__name__}

        if len(df) == 0:
            return False, {'error': 'Empty DataFrame'}

        # Check for numeric columns
        numeric_cols = list(df.select_dtypes(include='number').columns)

        if len(numeric_cols) < required_columns:
            return False, {
                'error': f'Too few numeric columns: {len(numeric_cols)} < {required_columns}',
                'numeric_columns': numeric_cols,
                'all_columns': list(df.columns),
            }

        # Check for nulls in numeric columns
        null_counts = df[numeric_cols].isnull().sum()
        if null_counts.any():
            cols_with_nulls = null_counts[null_counts > 0].to_dict()
            return False, {
                'error': 'Numeric columns contain nulls',
                'null_counts': cols_with_nulls,
            }

        # If allow_generated_params, try to find suitable columns
        if allow_generated_params:
            # Pick first two numeric columns
            x_col = numeric_cols[0]
            y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]

            params = {'points_x_by': x_col, 'points_y_by': y_col}

            # Add optional params if available
            if len(numeric_cols) > 2:
                params['point_size_by'] = numeric_cols[2]

            # Check for categorical columns for color
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                # Check if first categorical has reasonable cardinality
                first_cat = categorical_cols[0]
                if df[first_cat].nunique() < 50:
                    params['point_color_by'] = first_cat

            # Try actual cosmograph call if function provided
            if cosmo_function:
                try:
                    result = cosmo_function(df, **params)
                    return True, {
                        'params': params,
                        'result': result,
                        'note': 'Successfully created visualization',
                    }
                except Exception as e:
                    return False, {
                        'error': f'Cosmo call failed: {str(e)}',
                        'params': params,
                        'exception_type': type(e).__name__,
                    }
            else:
                # No actual cosmo function - just verify requirements met
                return True, {
                    'params': params,
                    'note': 'Requirements met (cosmo not called)',
                    'numeric_columns': numeric_cols,
                    'shape': df.shape,
                }

        # Basic validation passed
        return True, {
            'numeric_columns': numeric_cols,
            'shape': df.shape,
            'note': 'Basic requirements met',
        }

    return validate_cosmo_ready


def try_cosmo_visualization(
    df: Any, cosmo_function: Callable = None, **cosmo_kwargs
) -> tuple[bool, dict]:
    """Try to create a cosmograph visualization.

    This is a functional validator that actually attempts the visualization.

    Args:
        df: DataFrame to visualize
        cosmo_function: The cosmograph.cosmo function
        **cosmo_kwargs: Additional arguments for cosmo

    Returns:
        Tuple of (success, info)

    Example:
        >>> from cosmograph import cosmo  # doctest: +SKIP
        >>> success, info = try_cosmo_visualization(df, cosmo)  # doctest: +SKIP
    """
    if cosmo_function is None:
        try:
            from cosmograph import cosmo

            cosmo_function = cosmo
        except ImportError:
            return False, {
                'error': 'cosmograph not available',
                'note': 'Install with: pip install cosmograph',
            }

    # First check basic requirements
    validator = create_cosmo_validator(cosmo_function=None)
    basic_ok, basic_info = validator(df)

    if not basic_ok:
        return False, basic_info

    # Get suggested params if not provided
    params = cosmo_kwargs or basic_info.get('params', {})

    # Try the visualization
    try:
        result = cosmo_function(df, **params)
        return True, {
            'result': result,
            'params': params,
            'note': 'Visualization created successfully',
        }
    except Exception as e:
        import traceback

        return False, {
            'error': str(e),
            'exception_type': type(e).__name__,
            'traceback': traceback.format_exc(),
            'params': params,
        }


# Pre-configured validators for common use cases


def basic_cosmo_validator() -> Callable:
    """Basic validator that checks structural requirements only.

    Does not require cosmograph to be installed.

    Example:
        >>> validator = basic_cosmo_validator()  # doctest: +SKIP
        >>> success, info = validator(df)  # doctest: +SKIP
    """
    return create_cosmo_validator(cosmo_function=None, allow_generated_params=True)


def strict_cosmo_validator() -> Callable:
    """Strict validator that actually calls cosmograph.

    Requires cosmograph to be installed.

    Example:
        >>> validator = strict_cosmo_validator()  # doctest: +SKIP
        >>> success, info = validator(df)  # doctest: +SKIP
    """
    try:
        from cosmograph import cosmo

        return create_cosmo_validator(cosmo_function=cosmo, allow_generated_params=True)
    except ImportError:
        # Fall back to basic if cosmograph not available
        return basic_cosmo_validator()


def infer_cosmo_params(df) -> dict:
    """Infer suitable cosmograph parameters from DataFrame.

    Args:
        df: pandas DataFrame

    Returns:
        Dict of suggested parameters for cosmograph.cosmo()

    Example:
        >>> params = infer_cosmo_params(df)  # doctest: +SKIP
        >>> from cosmograph import cosmo  # doctest: +SKIP
        >>> cosmo(df, **params)  # doctest: +SKIP
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        return {}

    numeric_cols = list(df.select_dtypes(include='number').columns)

    if len(numeric_cols) < 2:
        return {}

    params = {'points_x_by': numeric_cols[0], 'points_y_by': numeric_cols[1]}

    # Add size if available
    if len(numeric_cols) > 2:
        params['point_size_by'] = numeric_cols[2]

    # Add color from categorical
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].nunique() < 50:  # Reasonable cardinality
            params['point_color_by'] = col
            break

    return params
