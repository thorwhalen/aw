"""Validation system supporting three flavors: schema, info-dict, and functional.

This module provides validators that can be used to check if artifacts
meet requirements. Validators follow the pattern:
    validator(artifact) -> (success: bool, info: dict)
"""

from typing import Any, Callable
from collections.abc import Mapping


def _ensure_validator_protocol(func: Callable) -> Callable[[Any], tuple[bool, dict]]:
    """Ensure a function returns (bool, dict) tuple.

    Wraps functions that only return bool to add empty info dict.
    """

    def wrapper(artifact):
        result = func(artifact)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        # If function only returns bool, add empty info dict
        return result, {}

    return wrapper


# ============================================================================
# Flavor 1: Schema-Based Validators
# ============================================================================


def schema_validator(schema: Any) -> Callable[[Any], tuple[bool, dict]]:
    """Create a validator from a schema (Pydantic, JSON Schema, etc.).

    Args:
        schema: A schema object with validation capability

    Returns:
        A validator function

    Example:
        >>> from pydantic import BaseModel
        >>> class DataSchema(BaseModel):
        ...     x: float
        ...     y: float
        >>> validate = schema_validator(DataSchema)
        >>> validate({'x': 1.0, 'y': 2.0})
        (True, {'validated': ...})
    """

    def validate(artifact):
        try:
            # Try Pydantic model validation
            if hasattr(schema, 'model_validate'):
                result = schema.model_validate(artifact)
                return True, {'validated': result}
            # Try JSON schema validation
            elif hasattr(schema, 'validate'):
                schema.validate(artifact)
                return True, {}
            else:
                return False, {'error': 'Unknown schema type'}
        except Exception as e:
            return False, {'error': str(e), 'exception_type': type(e).__name__}

    return validate


# ============================================================================
# Flavor 2: Info-Dict Based Validators
# ============================================================================


def info_dict_validator(
    compute_info: Callable[[Any], dict], check_info: Callable[[dict], tuple[bool, str]]
) -> Callable[[Any], tuple[bool, dict]]:
    """Create a validator that computes info then checks it.

    This is a two-stage validator:
    1. Compute information about the artifact (e.g., df.shape, df.dtypes)
    2. Check if that information meets requirements

    Args:
        compute_info: Function to extract info from artifact
        check_info: Function to check if info is acceptable

    Returns:
        A validator function

    Example:
        >>> def compute_df_info(df):
        ...     return {'shape': df.shape, 'null_count': df.isnull().sum().sum()}
        >>> def check_df_info(info):
        ...     if info['null_count'] > 10:
        ...         return False, "Too many nulls"
        ...     return True, "OK"
        >>> validate = info_dict_validator(compute_df_info, check_df_info)
    """

    def validate(artifact):
        try:
            info = compute_info(artifact)
            success, reason = check_info(info)
            return success, {'info': info, 'reason': reason}
        except Exception as e:
            return False, {'error': str(e), 'exception_type': type(e).__name__}

    return validate


# ============================================================================
# Flavor 3: Functional/Try Validators (Try the Purpose)
# ============================================================================


def functional_validator(
    try_function: Callable[[Any], Any], success_check: Callable[[Any], bool] = None
) -> Callable[[Any], tuple[bool, dict]]:
    """Create a validator that tries to use the artifact for its purpose.

    This validator actually attempts to use the artifact in the way it's
    intended to be used (e.g., try to visualize the data, try to train a model).

    Args:
        try_function: Function that tries to use the artifact
        success_check: Optional function to check if result is acceptable

    Returns:
        A validator function

    Example:
        >>> def try_cosmo(df):
        ...     return cosmograph.cosmo(df, points_x_by='x', points_y_by='y')
        >>> validate = functional_validator(try_cosmo)
        >>> validate(df)  # Tries to create viz, returns (True, {result}) or (False, {error})
    """

    def validate(artifact):
        try:
            result = try_function(artifact)
            # If success_check provided, use it; otherwise any result = success
            if success_check is not None:
                success = success_check(result)
            else:
                success = result is not None
            return success, {'result': result}
        except Exception as e:
            return False, {
                'error': str(e),
                'exception_type': type(e).__name__,
                'traceback': _get_traceback_str(e),
            }

    return validate


def _get_traceback_str(exception: Exception) -> str:
    """Extract traceback string from exception."""
    import traceback

    return ''.join(traceback.format_tb(exception.__traceback__))


# ============================================================================
# Composite Validators
# ============================================================================


def all_validators(*validators: Callable) -> Callable[[Any], tuple[bool, dict]]:
    """Combine multiple validators - all must pass.

    Args:
        *validators: Validator functions to combine

    Returns:
        A validator that passes only if all validators pass

    Example:
        >>> validate = all_validators(
        ...     is_dataframe,
        ...     has_required_columns(['x', 'y']),
        ...     has_no_nulls
        ... )
    """

    def validate(artifact):
        all_info = {}
        for i, validator in enumerate(validators):
            success, info = validator(artifact)
            all_info[f'validator_{i}'] = {'success': success, 'info': info}
            if not success:
                return False, all_info
        return True, all_info

    return validate


def any_validator(*validators: Callable) -> Callable[[Any], tuple[bool, dict]]:
    """Combine multiple validators - at least one must pass.

    Args:
        *validators: Validator functions to combine

    Returns:
        A validator that passes if any validator passes
    """

    def validate(artifact):
        all_info = {}
        for i, validator in enumerate(validators):
            success, info = validator(artifact)
            all_info[f'validator_{i}'] = {'success': success, 'info': info}
            if success:
                return True, all_info
        return False, all_info

    return validate


# ============================================================================
# Common Validators (Building Blocks)
# ============================================================================


def is_type(expected_type: type) -> Callable[[Any], tuple[bool, dict]]:
    """Validator that checks artifact type.

    Example:
        >>> import pandas as pd
        >>> validate = is_type(pd.DataFrame)
    """

    def validate(artifact):
        success = isinstance(artifact, expected_type)
        info = {
            'expected_type': expected_type.__name__,
            'actual_type': type(artifact).__name__,
        }
        return success, info

    return validate


def is_not_empty() -> Callable[[Any], tuple[bool, dict]]:
    """Validator that checks artifact is not empty.

    Works for sequences, mappings, DataFrames, etc.
    """

    def validate(artifact):
        try:
            is_empty = len(artifact) == 0
            return not is_empty, {'length': len(artifact)}
        except TypeError:
            # Object has no len - consider non-empty if it exists
            return artifact is not None, {'has_length': False}

    return validate


def has_attributes(**required_attrs: Any) -> Callable[[Any], tuple[bool, dict]]:
    """Validator that checks artifact has required attributes.

    Example:
        >>> validate = has_attributes(shape=lambda s: len(s) == 2, columns=lambda c: len(c) > 0)
    """

    def validate(artifact):
        info = {}
        for attr_name, checker in required_attrs.items():
            if not hasattr(artifact, attr_name):
                return False, {'missing_attribute': attr_name, 'info': info}
            attr_value = getattr(artifact, attr_name)
            if checker is not None:
                if callable(checker):
                    check_passed = checker(attr_value)
                else:
                    check_passed = attr_value == checker
                info[attr_name] = {'value': attr_value, 'check_passed': check_passed}
                if not check_passed:
                    return False, info
            else:
                info[attr_name] = {'value': attr_value}
        return True, info

    return validate
