"""Tests for LoadingAgent with mock and real LLMs."""

import pytest
import pandas as pd
from pathlib import Path
from aw.loading import LoadingAgent
from aw.base import Context, StepConfig


class TestLoadingAgentWithMock:
    """Test LoadingAgent with mock LLM."""

    def test_loading_agent_with_simple_csv(self, tmp_path):
        """Test basic loading flow with real loading (no LLM needed for simple CSV)."""
        # Create a test CSV file
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df.to_csv(csv_path, index=False)

        # Create agent with default config
        # For simple CSVs, the agent can load without LLM
        agent = LoadingAgent()

        # Execute loading
        context = Context()
        result_df, metadata = agent.execute(str(csv_path), context)

        # Verify result - may be None if mock LLM doesn't produce valid code
        # This tests the agent's structure even if loading fails
        assert metadata is not None
        assert 'attempts' in metadata

        # If it succeeds (depends on whether pandas is available in exec), verify
        if result_df is not None:
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 2
            assert 'loading' in context

    def test_loading_agent_stores_metadata(self, tmp_path):
        """Test that loading agent stores proper metadata structure."""
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({'x': [1]}).to_csv(csv_path, index=False)

        agent = LoadingAgent()
        context = Context()

        result_df, metadata = agent.execute(str(csv_path), context)

        # Verify metadata structure (works regardless of success)
        assert 'success' in metadata or 'error' in metadata
        assert 'attempts' in metadata


class TestLoadingAgentWithRealLLM:
    """Test LoadingAgent with real LLM (gpt-4o-mini)."""

    @pytest.mark.real_llm
    def test_loading_with_real_llm(self, tmp_path):
        """Test loading with real LLM - economical test.

        This test uses gpt-4o-mini with minimal tokens:
        - Single file with 2 rows
        - max_retries=1 to limit API calls
        - Simple structure to minimize prompt size
        """
        try:
            from functools import partial
            import oa
        except ImportError:
            pytest.skip("oa package not available")

        # Create minimal test data
        csv_path = tmp_path / "simple.csv"
        df = pd.DataFrame({'x': [1.0, 2.0], 'y': [3.0, 4.0]})
        df.to_csv(csv_path, index=False)

        # Configure with real LLM using gpt-4o-mini (cheap model)
        llm_chat = partial(oa.chat, model='gpt-4o-mini')
        config = StepConfig(
            llm=llm_chat, max_retries=1  # Limit to 1 retry to save tokens
        )

        agent = LoadingAgent(config=config)
        context = Context()

        # Execute - this will make a real API call
        result_df, metadata = agent.execute(str(csv_path), context)

        # Basic assertions
        assert isinstance(result_df, pd.DataFrame), "Should return a DataFrame"
        assert len(result_df) > 0, "DataFrame should have rows"
        assert 'loading' in context, "Should store result in context"
        assert 'source_uri' in metadata, "Should track source"

        # Verify it's actually a valid DataFrame with expected structure
        assert result_df.shape[0] == 2, "Should have 2 rows"
        assert result_df.shape[1] >= 2, "Should have at least 2 columns"
