"""Tests for validation system."""

import pytest
import pandas as pd
from aw.validation import (
    schema_validator,
    info_dict_validator,
    functional_validator,
    all_validators,
    is_type,
    has_attributes,
)


class TestSchemaValidator:
    """Test schema-based validation."""

    def test_dict_validation(self):
        """Test validating a dict against a simple schema."""
        from pydantic import BaseModel

        class Point(BaseModel):
            x: float
            y: float

        validate = schema_validator(Point)
        valid, result = validate({'x': 1.0, 'y': 2.0})

        assert valid is True
        # schema_validator returns {'validated': model_instance}
        assert 'validated' in result
        assert result['validated'].x == 1.0
        assert result['validated'].y == 2.0

    def test_schema_validation_failure(self):
        """Test schema validation fails on invalid data."""
        from pydantic import BaseModel

        class Point(BaseModel):
            x: float
            y: float

        validate = schema_validator(Point)
        valid, result = validate({'x': 'not_a_number', 'y': 2.0})

        assert valid is False
        assert 'error' in result


class TestInfoDictValidator:
    """Test info-dict validation."""

    def test_compute_and_check(self, sample_df):
        """Test validation using compute→check pattern."""

        def compute_info(df):
            return {'n_rows': len(df), 'n_cols': len(df.columns)}

        def check_info(info):
            if info['n_rows'] < 1:
                return False, {'error': 'Need at least one row'}
            return True, info

        validate = info_dict_validator(compute_info, check_info)
        valid, result = validate(sample_df)

        assert valid is True
        # info_dict_validator returns {'info': ..., 'reason': ...}
        assert 'info' in result
        assert result['info']['n_rows'] == 4
        assert result['info']['n_cols'] == 3

    def test_info_dict_validation_failure(self, sample_df):
        """Test info-dict validation can fail."""

        def compute_info(df):
            return {'n_rows': len(df)}

        def check_info(info):
            if info['n_rows'] < 10:
                return False, {'error': 'Need at least 10 rows'}
            return True, info

        validate = info_dict_validator(compute_info, check_info)
        valid, result = validate(sample_df)

        assert valid is False
        # info_dict_validator returns {'info': ..., 'reason': ...}
        assert 'reason' in result
        assert result['reason']['error'] == 'Need at least 10 rows'


class TestFunctionalValidator:
    """Test functional validation."""

    def test_successful_function_execution(self, sample_df):
        """Test validation passes when function succeeds."""

        def process(df):
            return df.sum()

        validate = functional_validator(process)
        valid, result = validate(sample_df)

        assert valid is True
        assert 'result' in result

    def test_function_execution_failure(self, sample_df):
        """Test validation fails when function raises."""

        def process(df):
            raise ValueError("Something went wrong")

        validate = functional_validator(process)
        valid, result = validate(sample_df)

        assert valid is False
        assert 'error' in result


class TestValidatorCombinators:
    """Test validator composition."""

    def test_is_type_validator(self, sample_df):
        """Test type checking validator."""
        validate = is_type(pd.DataFrame)
        valid, result = validate(sample_df)

        assert valid is True

    def test_is_type_fails_on_wrong_type(self):
        """Test type validator fails on wrong type."""
        validate = is_type(pd.DataFrame)
        valid, result = validate([1, 2, 3])

        assert valid is False

    def test_has_attributes_validator(self, sample_df):
        """Test attribute checking validator."""
        validate = has_attributes(
            shape=lambda s: len(s) == 2, columns=lambda c: len(c) > 0
        )
        valid, result = validate(sample_df)

        assert valid is True

    def test_all_validators_composition(self, sample_df):
        """Test combining multiple validators."""

        def has_x_column(df):
            if 'x' not in df.columns:
                return False, {'error': 'Missing x column'}
            return True, {}

        def has_enough_rows(df):
            if len(df) < 2:
                return False, {'error': 'Need at least 2 rows'}
            return True, {}

        validate = all_validators(has_x_column, has_enough_rows)
        valid, result = validate(sample_df)

        assert valid is True

    def test_all_validators_fails_on_any_failure(self, sample_df):
        """Test all_validators fails if any validator fails."""

        def always_pass(df):
            return True, {}

        def always_fail(df):
            return False, {'error': 'Forced failure'}

        validate = all_validators(always_pass, always_fail)
        valid, result = validate(sample_df)

        assert valid is False
        # all_validators returns {validator_0: ..., validator_1: ...}
        assert 'validator_1' in result
        assert result['validator_1']['success'] is False
        assert result['validator_1']['info']['error'] == 'Forced failure'
