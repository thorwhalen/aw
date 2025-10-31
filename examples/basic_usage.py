"""Example usage of the aw package for data preparation.

This script demonstrates:
1. Basic loading and preparation
2. Using validators
3. Creating workflows
4. Custom configurations
"""

import pandas as pd
from pathlib import Path


def create_sample_data():
    """Create sample CSV data for testing."""
    # Create sample data
    data = {
        'x': [1, 2, 3, 4, 5, None, 7, 8],
        'y': [2.5, 3.1, 4.2, 5.3, 6.1, 7.2, 8.1, 9.0],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'value': [10, 20, 15, 25, 30, 35, 40, 45],
    }
    df = pd.DataFrame(data)

    # Save to temp file
    temp_path = Path('/tmp/sample_data.csv')
    df.to_csv(temp_path, index=False)
    print(f"✓ Created sample data at {temp_path}")
    return str(temp_path)


def example_1_basic_usage():
    """Example 1: Basic loading agent usage."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Loading Agent")
    print("=" * 60)

    from aw import LoadingAgent, Context

    # Create sample data
    data_path = create_sample_data()

    # Create agent and context
    agent = LoadingAgent()
    context = Context()

    # Load data
    df, metadata = agent.execute(data_path, context)

    print(f"\n✓ Loaded DataFrame: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Attempts: {metadata['num_attempts']}")
    print(f"\nFirst few rows:")
    print(df.head())


def example_2_preparation():
    """Example 2: Preparation agent for cosmograph."""
    print("\n" + "=" * 60)
    print("Example 2: Preparation Agent")
    print("=" * 60)

    from aw import LoadingAgent, PreparationAgent, Context
    from aw.cosmo import basic_cosmo_validator

    data_path = create_sample_data()
    context = Context()

    # Load
    loading_agent = LoadingAgent()
    df, _ = loading_agent.execute(data_path, context)
    print(f"✓ Loaded: {df.shape}")

    # Prepare
    preparing_agent = PreparationAgent(
        target='cosmo-ready', target_validator=basic_cosmo_validator()
    )
    prepared_df, prep_meta = preparing_agent.execute(df, context)

    print(f"✓ Prepared: {prepared_df.shape}")
    print(f"  Validation: {prep_meta['validation_result']}")

    if prep_meta['success']:
        params = prep_meta['validation_result'].get('params', {})
        print(f"\n  Suggested cosmo params:")
        for key, val in params.items():
            print(f"    {key}: {val}")


def example_3_workflow():
    """Example 3: Complete workflow."""
    print("\n" + "=" * 60)
    print("Example 3: Complete Workflow")
    print("=" * 60)

    from aw import create_cosmo_prep_workflow

    data_path = create_sample_data()

    # Create and run workflow
    workflow = create_cosmo_prep_workflow(max_retries=3)
    result_df, metadata = workflow.run(data_path)

    if metadata.get('success'):
        print(f"✓ Workflow completed successfully!")
        print(f"  Steps executed: {[s['name'] for s in metadata['steps']]}")
        print(f"  Final shape: {result_df.shape}")
    else:
        print(f"✗ Workflow failed at: {metadata.get('failed_at')}")


def example_4_validation():
    """Example 4: Different validation approaches."""
    print("\n" + "=" * 60)
    print("Example 4: Validation Flavors")
    print("=" * 60)

    import pandas as pd
    from aw.validation import (
        info_dict_validator,
        functional_validator,
        all_validators,
        is_type,
        is_not_empty,
    )

    # Create sample dataframe
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

    # 1. Info-dict validator
    print("\n1. Info-dict Validator:")

    def compute_info(df):
        return {
            'shape': df.shape,
            'numeric_cols': len(df.select_dtypes(include='number').columns),
        }

    def check_info(info):
        if info['numeric_cols'] < 2:
            return False, "Need at least 2 numeric columns"
        return True, "OK"

    validator = info_dict_validator(compute_info, check_info)
    success, info = validator(df)
    print(f"  Success: {success}")
    print(f"  Info: {info}")

    # 2. Functional validator
    print("\n2. Functional Validator:")

    def try_mean(df):
        return df.mean()

    validator = functional_validator(try_mean)
    success, info = validator(df)
    print(f"  Success: {success}")
    print(f"  Result: {info.get('result')}")

    # 3. Composite validator
    print("\n3. Composite Validator:")

    validator = all_validators(is_type(pd.DataFrame), is_not_empty())
    success, info = validator(df)
    print(f"  Success: {success}")


def example_5_custom_config():
    """Example 5: Custom configuration."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Configuration")
    print("=" * 60)

    from aw import GlobalConfig, LoadingAgent, Context

    # Create global config
    global_config = GlobalConfig(llm="gpt-4", max_retries=5)

    # Create step config with override
    step_config = global_config.override(max_retries=3)  # Override just this

    # Use with agent
    agent = LoadingAgent(config=step_config)

    print(f"✓ Created agent with config:")
    print(f"  LLM: {step_config.llm}")
    print(f"  Max retries: {step_config.max_retries}")


def example_6_code_interpreter():
    """Example 6: Code interpreter tool."""
    print("\n" + "=" * 60)
    print("Example 6: Code Interpreter")
    print("=" * 60)

    from aw import CodeInterpreterTool

    tool = CodeInterpreterTool()

    # Execute some code
    code = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
result = df.mean()
print(result)
"""

    result = tool(code)

    print(f"✓ Code execution:")
    print(f"  Success: {result.success}")
    print(f"  Output:\n{result.output}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("AW Package Examples")
    print("=" * 60)

    try:
        example_1_basic_usage()
        example_2_preparation()
        example_3_workflow()
        example_4_validation()
        example_5_custom_config()
        example_6_code_interpreter()

        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == '__main__':
    main()
