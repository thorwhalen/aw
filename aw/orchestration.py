"""Orchestration and workflow management for agentic steps.

Provides utilities to chain multiple agentic steps together and manage
the overall workflow.
"""

from typing import Any, Callable, Union
from collections.abc import MutableMapping

from aw.base import AgenticStep, Context, StepConfig
from aw.loading import LoadingAgent
from aw.preparing import PreparationAgent


class AgenticWorkflow:
    """Orchestrates a chain of agentic steps.

    Manages the execution of multiple steps in sequence, handling
    context/artifact passing between steps.

    Example:
        >>> workflow = AgenticWorkflow()
        >>> workflow.add_step('loading', loading_agent)
        >>> workflow.add_step('preparing', preparing_agent)
        >>> result = workflow.run(source_uri)
    """

    def __init__(self, context: Context = None):
        """Initialize workflow.

        Args:
            context: Shared context for all steps (creates new if None)
        """
        self.context = context or Context()
        self.steps: list[tuple[str, AgenticStep]] = []

    def add_step(self, name: str, step: AgenticStep) -> 'AgenticWorkflow':
        """Add a step to the workflow.

        Args:
            name: Name/identifier for the step
            step: Agent or step implementation

        Returns:
            Self for chaining
        """
        self.steps.append((name, step))
        return self

    def run(self, initial_input: Any) -> tuple[Any, dict[str, Any]]:
        """Execute the workflow.

        Args:
            initial_input: Input to the first step

        Returns:
            Tuple of (final_artifact, workflow_metadata)
        """
        current_input = initial_input
        workflow_metadata = {'steps': [], 'context_snapshot': None}

        for step_name, step in self.steps:
            # Execute step
            artifact, metadata = step.execute(current_input, self.context)

            # Record step execution
            workflow_metadata['steps'].append(
                {
                    'name': step_name,
                    'success': metadata.get('success', True),
                    'metadata': metadata,
                }
            )

            # Check for failure
            if not metadata.get('success', True):
                workflow_metadata['failed_at'] = step_name
                workflow_metadata['context_snapshot'] = self.context.snapshot()
                return artifact, workflow_metadata

            # Pass artifact to next step
            current_input = artifact

        # All steps completed successfully
        workflow_metadata['success'] = True
        workflow_metadata['context_snapshot'] = self.context.snapshot()

        return current_input, workflow_metadata

    def run_partial(
        self, initial_input: Any, stop_after: str = None
    ) -> tuple[Any, dict[str, Any]]:
        """Run workflow up to a specific step.

        Args:
            initial_input: Input to first step
            stop_after: Name of step to stop after (runs all if None)

        Returns:
            Tuple of (artifact, metadata)
        """
        current_input = initial_input
        workflow_metadata = {'steps': []}

        for step_name, step in self.steps:
            artifact, metadata = step.execute(current_input, self.context)

            workflow_metadata['steps'].append({'name': step_name, 'metadata': metadata})

            current_input = artifact

            if step_name == stop_after:
                break

        workflow_metadata['context_snapshot'] = self.context.snapshot()
        return current_input, workflow_metadata


def create_data_prep_workflow(
    loading_config: StepConfig = None,
    preparing_config: StepConfig = None,
    target: str = 'generic',
) -> AgenticWorkflow:
    """Factory to create a data preparation workflow.

    Creates a workflow with LoadingAgent -> PreparationAgent.

    Args:
        loading_config: Configuration for loading agent
        preparing_config: Configuration for preparing agent
        target: Target format for preparation

    Returns:
        Configured workflow

    Example:
        >>> workflow = create_data_prep_workflow(target='cosmo-ready')
        >>> df, metadata = workflow.run('/path/to/data.csv')
    """
    workflow = AgenticWorkflow()

    # Add loading step
    loading_agent = LoadingAgent(loading_config)
    workflow.add_step('loading', loading_agent)

    # Add preparing step
    preparing_agent = PreparationAgent(config=preparing_config, target=target)
    workflow.add_step('preparing', preparing_agent)

    return workflow


def create_cosmo_prep_workflow(
    cosmo_validator: Callable = None, max_retries: int = 3
) -> AgenticWorkflow:
    """Create a workflow specifically for cosmograph preparation.

    Args:
        cosmo_validator: Custom cosmo validator (uses default if None)
        max_retries: Maximum retries for each step

    Returns:
        Configured workflow

    Example:
        >>> workflow = create_cosmo_prep_workflow()
        >>> prepared_df, metadata = workflow.run('data.csv')
    """
    from aw.cosmo import basic_cosmo_validator

    # Use provided validator or default
    validator = cosmo_validator or basic_cosmo_validator()

    # Create configs
    loading_config = StepConfig(max_retries=max_retries)
    preparing_config = StepConfig(max_retries=max_retries, validator=validator)

    return create_data_prep_workflow(
        loading_config=loading_config,
        preparing_config=preparing_config,
        target='cosmo-ready',
    )


# ============================================================================
# Convenience Functions
# ============================================================================


def load_and_prepare(
    source_uri: str,
    target: str = 'generic',
    validator: Callable = None,
    max_retries: int = 3,
) -> tuple[Any, dict]:
    """Convenience function to load and prepare data in one call.

    Args:
        source_uri: URI of data source
        target: Target format/purpose
        validator: Optional custom validator
        max_retries: Maximum retry attempts

    Returns:
        Tuple of (prepared_dataframe, metadata)

    Example:
        >>> df, meta = load_and_prepare('data.csv', target='cosmo-ready')
    """
    loading_config = StepConfig(max_retries=max_retries)
    preparing_config = StepConfig(max_retries=max_retries, validator=validator)

    workflow = create_data_prep_workflow(
        loading_config=loading_config, preparing_config=preparing_config, target=target
    )

    return workflow.run(source_uri)


def load_for_cosmo(
    source_uri: str, max_retries: int = 3, strict: bool = False
) -> tuple[Any, dict]:
    """Load and prepare data specifically for cosmograph visualization.

    Args:
        source_uri: URI of data source
        max_retries: Maximum retry attempts
        strict: If True, actually calls cosmograph for validation

    Returns:
        Tuple of (prepared_dataframe, metadata)

    Example:
        >>> df, meta = load_for_cosmo('data.csv')
        >>> from cosmograph import cosmo
        >>> cosmo(df, **meta['preparing']['metadata']['validation_result']['params'])
    """
    from aw.cosmo import basic_cosmo_validator, strict_cosmo_validator

    validator = strict_cosmo_validator() if strict else basic_cosmo_validator()

    return load_and_prepare(
        source_uri=source_uri,
        target='cosmo-ready',
        validator=validator,
        max_retries=max_retries,
    )


# ============================================================================
# Interactive Workflow (with Human-in-Loop)
# ============================================================================


class InteractiveWorkflow(AgenticWorkflow):
    """Workflow with human-in-the-loop capabilities.

    Pauses execution to request human input/approval at configured points.

    Example:
        >>> workflow = InteractiveWorkflow()
        >>> workflow.add_step('loading', agent, require_approval=True)
        >>> result = workflow.run_interactive(source_uri)
    """

    def __init__(self, context: Context = None):
        super().__init__(context)
        self.approval_required: dict[str, bool] = {}
        self.approval_callback: Callable = None

    def add_step(
        self, name: str, step: AgenticStep, require_approval: bool = False
    ) -> 'InteractiveWorkflow':
        """Add step with optional approval requirement."""
        super().add_step(name, step)
        self.approval_required[name] = require_approval
        return self

    def set_approval_callback(self, callback: Callable[[str, Any, dict], bool]) -> None:
        """Set callback for human approval.

        Args:
            callback: Function(step_name, artifact, metadata) -> bool
                Returns True to continue, False to abort
        """
        self.approval_callback = callback

    def run_interactive(self, initial_input: Any) -> tuple[Any, dict]:
        """Run workflow with human-in-loop checkpoints."""
        current_input = initial_input
        workflow_metadata = {'steps': []}

        for step_name, step in self.steps:
            # Execute step
            artifact, metadata = step.execute(current_input, self.context)

            workflow_metadata['steps'].append({'name': step_name, 'metadata': metadata})

            # Check if approval required
            if self.approval_required.get(step_name, False):
                if self.approval_callback:
                    approved = self.approval_callback(step_name, artifact, metadata)
                    if not approved:
                        workflow_metadata['aborted_at'] = step_name
                        workflow_metadata['reason'] = 'User rejected'
                        return artifact, workflow_metadata
                else:
                    # No callback - just print and ask
                    print(f"\nStep '{step_name}' completed:")
                    print(f"  Metadata: {metadata}")
                    response = input("Continue? (y/n): ")
                    if response.lower() != 'y':
                        workflow_metadata['aborted_at'] = step_name
                        return artifact, workflow_metadata

            current_input = artifact

        workflow_metadata['success'] = True
        workflow_metadata['context_snapshot'] = self.context.snapshot()
        return current_input, workflow_metadata
