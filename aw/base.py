"""Core protocols and base classes for agentic workflows.

This module provides the foundational abstractions for building AI agents
that follow the AgenticStep protocol with ReAct pattern (Reason-Act-Observe).
"""

from typing import Protocol, Any, Callable, Union, TypeVar, Generic
from dataclasses import dataclass, field
from collections.abc import Mapping, MutableMapping


T = TypeVar('T')
ArtifactType = TypeVar('ArtifactType')


class AgenticStep(Protocol[ArtifactType]):
    """Protocol defining the interface for any agentic step in a workflow.

    Each step follows the ReAct pattern:
    1. Reason (Thought): Analyze input and context
    2. Act (Action): Generate and execute code or use tools
    3. Observe: Capture results and validate
    4. Repeat or finish based on validation

    Example:
        >>> class MyAgent:
        ...     def execute(self, input_data, context):
        ...         # Agent implementation
        ...         return result, metadata
    """

    def execute(
        self, input_data: Any, context: MutableMapping[str, Any]
    ) -> tuple[ArtifactType, dict[str, Any]]:
        """Execute the agentic step.

        Args:
            input_data: The input to process
            context: Mutable mapping containing shared state and history

        Returns:
            Tuple of (artifact, metadata) where artifact is the main output
            and metadata contains auxiliary information
        """
        ...


class Validator(Protocol):
    """Protocol for validation functions.

    Validators check if an artifact meets requirements and return
    both a success indicator and detailed information.

    Example:
        >>> def is_non_empty_dataframe(df):
        ...     success = df is not None and len(df) > 0
        ...     info = {'shape': df.shape if success else None}
        ...     return success, info
    """

    def __call__(self, artifact: Any) -> tuple[bool, dict[str, Any]]:
        """Validate an artifact.

        Args:
            artifact: The artifact to validate

        Returns:
            Tuple of (success, info) where success is True if validation
            passed and info contains details about the validation
        """
        ...


class Tool(Protocol):
    """Protocol for tools that agents can use.

    Tools are callable objects that perform specific actions like
    executing code, sampling files, or calling external APIs.

    Example:
        >>> class FileSampler:
        ...     def __call__(self, uri):
        ...         # Sample file logic
        ...         return {'extension': '.csv', 'sample': '...'}
    """

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool."""
        ...


@dataclass
class StepConfig:
    """Configuration for an AgenticStep.

    Supports both simple objects (strings, ints) and callables for maximum
    flexibility with dependency injection.

    Attributes:
        llm: Either a model name string or a text-to-text chat function
        validator: Validation callable or schema object
        tools: List of callable tools available to the agent
        max_retries: Maximum number of retry attempts
        human_in_loop: Whether to require human approval/intervention

    Example:
        >>> config = StepConfig(
        ...     llm="gpt-4",
        ...     validator=lambda x: (True, {}),
        ...     max_retries=3
        ... )
    """

    llm: Union[str, Callable[[str], str]] = "gpt-4"
    validator: Union[Callable, Any] = None
    tools: list[Callable] = field(default_factory=list)
    max_retries: int = 3
    human_in_loop: bool = False

    def resolve_llm(self) -> Callable[[str], str]:
        """Resolve llm to a callable function.

        Returns:
            A callable that takes a prompt string and returns a response string
        """
        if callable(self.llm):
            return self.llm
        # Default implementation - will be replaced by actual LLM
        return lambda prompt: f"Response to: {prompt}"

    def resolve_validator(self) -> Validator:
        """Resolve validator to a callable.

        Returns:
            A callable that validates artifacts
        """
        if self.validator is None:
            # Default validator - always passes
            return lambda artifact: (True, {})
        if callable(self.validator):
            return self.validator
        # Handle schema-based validators (Pydantic, etc.)
        return self._schema_to_validator(self.validator)

    def _schema_to_validator(self, schema: Any) -> Validator:
        """Convert a schema object to a validator function."""

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
                    # Unknown schema type - pass through
                    return True, {}
            except Exception as e:
                return False, {'error': str(e)}

        return validate


@dataclass
class GlobalConfig:
    """Global configuration with cascading defaults.

    Provides defaults that can be overridden at step or agent level.

    Example:
        >>> global_cfg = GlobalConfig(llm="gpt-4", max_retries=5)
        >>> step_cfg = global_cfg.override(llm="gpt-3.5-turbo")
    """

    llm: Union[str, Callable[[str], str]] = "gpt-4"
    max_retries: int = 3
    human_in_loop: bool = False

    def override(self, **kwargs) -> StepConfig:
        """Create a StepConfig with overridden values.

        Args:
            **kwargs: Values to override from global defaults

        Returns:
            A new StepConfig with merged configuration
        """
        defaults = {
            'llm': self.llm,
            'max_retries': self.max_retries,
            'human_in_loop': self.human_in_loop,
        }
        defaults.update(kwargs)
        return StepConfig(**defaults)


class Context(MutableMapping[str, Any]):
    """Context for sharing state and artifacts between agentic steps.

    Implements MutableMapping interface for dict-like behavior while
    providing additional functionality for managing agent state.

    Example:
        >>> ctx = Context()
        >>> ctx['loading'] = {'df': df, 'info': {...}}
        >>> ctx['preparing'] = {'df': prepared_df}
    """

    def __init__(self, initial_data: dict = None):
        self._data: dict[str, Any] = initial_data or {}
        self._history: list[tuple[str, Any]] = []

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._history.append((key, value))
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def history(self) -> list[tuple[str, Any]]:
        """Access the history of all context updates."""
        return self._history

    def snapshot(self) -> dict[str, Any]:
        """Create a snapshot of current context state.

        Returns:
            A dict copy of current context data
        """
        return dict(self._data)
