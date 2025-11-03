"""Shared fixtures for tests."""

import pytest
import pandas as pd


@pytest.fixture
def sample_df():
    """Create a simple DataFrame for testing."""
    return pd.DataFrame(
        {
            'x': [1.0, 2.0, 3.0, 4.0],
            'y': [2.0, 4.0, 6.0, 8.0],
            'label': ['a', 'b', 'c', 'd'],
        }
    )


@pytest.fixture
def mock_llm_echo():
    """Mock LLM that echoes prompts back."""

    def llm(prompt: str, **kwargs) -> str:
        return f"ECHO: {prompt}"

    return llm


@pytest.fixture
def mock_llm_code_gen():
    """Mock LLM that generates simple code."""

    def llm(prompt: str, **kwargs) -> str:
        if 'rename' in prompt.lower():
            return "df.rename(columns={'old': 'new'})"
        elif 'drop' in prompt.lower():
            return "df.dropna()"
        else:
            return "df"

    return llm
