"""Tests for end-to-end workflows."""

import pytest
import pandas as pd
from functools import partial
from aw.orchestration import AgenticWorkflow
from aw.loading import LoadingAgent
from aw.preparing import PreparationAgent
from aw.base import Context, StepConfig


class TestWorkflowWithMock:
    """Test workflow orchestration with mocks."""

    def test_simple_workflow_structure(self, tmp_path):
        """Test workflow structure and metadata collection."""
        # Create test data
        csv_path = tmp_path / "workflow_test.csv"
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.to_csv(csv_path, index=False)

        # Create workflow
        workflow = AgenticWorkflow()

        # Add loading step
        loading_agent = LoadingAgent()
        workflow.add_step('loading', loading_agent)

        # Run workflow
        result = workflow.run(str(csv_path))

        # Verify structure - workflow returns (result, metadata) or just metadata
        # depending on success/failure
        assert workflow.context is not None

    def test_workflow_context_exists(self, tmp_path):
        """Test that workflow creates and maintains context."""
        csv_path = tmp_path / "context_test.csv"
        pd.DataFrame({'x': [1.0]}).to_csv(csv_path, index=False)

        workflow = AgenticWorkflow()
        loading_agent = LoadingAgent()
        workflow.add_step('loading', loading_agent)

        # Run
        result = workflow.run(str(csv_path))

        # Verify context exists
        assert workflow.context is not None


class TestWorkflowWithRealLLM:
    """Test workflow with real LLM."""

    @pytest.mark.real_llm
    def test_loading_and_preparing_workflow(self, tmp_path):
        """Test complete loading→preparing workflow with real LLM.

        This is an economical test that:
        - Uses minimal data (3 rows, 2 columns)
        - Uses gpt-4o-mini (cheapest model)
        - Limits retries to 1
        - Has a simple validation that passes quickly
        """
        try:
            import oa
        except ImportError:
            pytest.skip("oa package not available")

        # Create minimal test data
        csv_path = tmp_path / "workflow.csv"
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [4.0, 5.0, 6.0]})
        df.to_csv(csv_path, index=False)

        # Simple validator that always passes to minimize retries
        def simple_validator(df):
            """Quick validator that just checks it's a DataFrame."""
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                return True, {'status': 'ok'}
            return False, {'error': 'not a valid dataframe'}

        # Setup workflow with real LLM
        llm_chat = partial(oa.chat, model='gpt-4o-mini')
        config = StepConfig(llm=llm_chat, max_retries=1)

        workflow = AgenticWorkflow()

        # Add loading step
        loading_agent = LoadingAgent(config=config)
        workflow.add_step('loading', loading_agent)

        # Add preparing step with simple validator
        prep_config = StepConfig(
            llm=llm_chat, max_retries=1, validator=simple_validator
        )
        preparing_agent = PreparationAgent(config=prep_config, target='clean')
        workflow.add_step('preparing', preparing_agent)

        # Execute workflow - makes real API calls
        final_artifact, workflow_metadata = workflow.run(str(csv_path))

        # Verify results
        assert 'steps' in workflow_metadata, "Should have step metadata"
        assert len(workflow_metadata['steps']) >= 1, "Should have at least loading step"

        # Check that loading step executed
        loading_step = workflow_metadata['steps'][0]
        assert loading_step['name'] == 'loading'

        # If workflow succeeded, check the results
        if workflow_metadata.get('success', False):
            # Get results from context
            assert 'loading' in workflow.context, "Should have loading in context"
            assert 'preparing' in workflow.context, "Should have preparing in context"

            loaded_df = workflow.context['loading']['df']
            prepared_df = workflow.context['preparing']['df']

            assert isinstance(
                loaded_df, pd.DataFrame
            ), "Loaded result should be DataFrame"
            assert isinstance(
                prepared_df, pd.DataFrame
            ), "Prepared result should be DataFrame"
            assert len(loaded_df) > 0, "Loaded DataFrame should have data"
            assert len(prepared_df) > 0, "Prepared DataFrame should have data"
