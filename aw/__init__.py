"""aw - Agentic Workflows for Data Preparation

An AI agent package for data preparation with a focus on loading and preparing
data for various purposes (e.g., visualization with cosmograph).

Key Features:
- AgenticStep protocol for building modular agents
- ReAct pattern (Reason-Act-Observe) with retry logic
- Three validation flavors: schema, info-dict, and functional
- Code execution with safe defaults and extensibility
- Loading and Preparation agents for data workflows
- Cosmograph-specific validators and utilities
- Orchestration for chaining multiple steps

Example:
    >>> from aw import load_for_cosmo
    >>> df, metadata = load_for_cosmo('data.csv')
    >>> from cosmograph import cosmo
    >>> params = metadata['preparing']['metadata']['validation_result']['params']
    >>> cosmo(df, **params)

Architecture:
    - aw.base: Core protocols and configurations
    - aw.validation: Three validation flavors
    - aw.tools: Code execution and agent tools
    - aw.utils: Helper functions and facades
    - aw.loading: LoadingAgent for data ingestion
    - aw.preparing: PreparationAgent for data transformation
    - aw.cosmo: Cosmograph-specific validators
    - aw.orchestration: Workflow management
"""

# Core abstractions
from aw.base import (
    AgenticStep,
    Validator,
    Tool,
    StepConfig,
    GlobalConfig,
    Context,
)

# Validation
from aw.validation import (
    schema_validator,
    info_dict_validator,
    functional_validator,
    all_validators,
    any_validator,
    is_type,
    is_not_empty,
    has_attributes,
)

# Tools
from aw.tools import (
    CodeInterpreterTool,
    SafeCodeInterpreter,
    ExecutionResult,
    create_langchain_executor,
)

# Utilities
from aw.utils import (
    FileSamplerTool,
    infer_loader_from_extension,
    infer_loader_params,
    compute_dataframe_info,
    get_numeric_columns,
    default_llm_factory,
    create_openai_chat,
    create_oa_chat,
)

# Agents
from aw.loading import LoadingAgent, create_loading_agent
from aw.preparing import PreparationAgent, create_preparation_agent

# Cosmograph support
from aw.cosmo import (
    create_cosmo_validator,
    try_cosmo_visualization,
    basic_cosmo_validator,
    strict_cosmo_validator,
    infer_cosmo_params,
)

# Orchestration
from aw.orchestration import (
    AgenticWorkflow,
    InteractiveWorkflow,
    create_data_prep_workflow,
    create_cosmo_prep_workflow,
    load_and_prepare,
    load_for_cosmo,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    'AgenticStep',
    'Validator',
    'Tool',
    'StepConfig',
    'GlobalConfig',
    'Context',
    # Validation
    'schema_validator',
    'info_dict_validator',
    'functional_validator',
    'all_validators',
    'any_validator',
    'is_type',
    'is_not_empty',
    'has_attributes',
    # Tools
    'CodeInterpreterTool',
    'SafeCodeInterpreter',
    'ExecutionResult',
    'create_langchain_executor',
    # Utils
    'FileSamplerTool',
    'infer_loader_from_extension',
    'infer_loader_params',
    'compute_dataframe_info',
    'get_numeric_columns',
    'default_llm_factory',
    'create_openai_chat',
    'create_oa_chat',
    # Agents
    'LoadingAgent',
    'create_loading_agent',
    'PreparationAgent',
    'create_preparation_agent',
    # Cosmo
    'create_cosmo_validator',
    'try_cosmo_visualization',
    'basic_cosmo_validator',
    'strict_cosmo_validator',
    'infer_cosmo_params',
    # Orchestration
    'AgenticWorkflow',
    'InteractiveWorkflow',
    'create_data_prep_workflow',
    'create_cosmo_prep_workflow',
    'load_and_prepare',
    'load_for_cosmo',
]
