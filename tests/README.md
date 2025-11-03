# AW Package Tests

This directory contains tests for the `aw` (Agentic Workflow) package.

## Test Structure

- `conftest.py` - Shared fixtures for tests
- `test_validation.py` - Tests for validation system (schema, info-dict, functional)
- `test_loading.py` - Tests for LoadingAgent (with mocks and real LLM)
- `test_workflow.py` - Tests for end-to-end workflow orchestration

## Running Tests

### Run all tests (including real LLM tests):
```bash
pytest tests/ -v
```

### Run only mock tests (no API calls):
```bash
pytest tests/ -v -m "not real_llm"
```

### Run only real LLM tests (makes API calls):
```bash
pytest tests/ -v -m "real_llm"
```

### Run with coverage:
```bash
pytest tests/ --cov=aw --cov-report=html
```

### Run doctests:
```bash
pytest --doctest-modules aw/ -v
```

### Run everything (doctests + unit tests):
```bash
pytest --doctest-modules aw/ tests/ -v
```

## Test Categories

### Mock Tests (No LLM API Calls)

These tests use mock LLMs and focus on testing the structure and logic:

- **Validation Tests**: All 13 validation tests use no LLM
  - Schema validation (Pydantic)
  - Info-dict validation (compute→check pattern)
  - Functional validation (try-catch pattern)
  - Validator combinators (is_type, has_attributes, all_validators)

- **Loading Tests (Mock)**: 2 tests
  - Basic agent structure
  - Metadata collection

- **Workflow Tests (Mock)**: 2 tests
  - Workflow orchestration structure
  - Context passing between steps

### Real LLM Tests (API Calls)

These tests make actual API calls to OpenAI using `gpt-4o-mini` (economical model):

- **test_loading_with_real_llm**: Tests LoadingAgent with real LLM
  - Creates minimal CSV (2 rows, 2 columns)
  - Uses `oa.chat` with `model='gpt-4o-mini'`
  - Limits to 1 retry to save tokens
  - Verifies actual DataFrame loading

- **test_loading_and_preparing_workflow**: Tests complete workflow
  - Creates minimal CSV (3 rows, 2 columns)
  - Runs loading→preparing pipeline
  - Uses simple validator to minimize retries
  - Limits to 1 retry per step
  - Verifies end-to-end agentic workflow

## Test Philosophy

1. **Mock First**: Most tests (15/17) use mocks to test logic without API costs
2. **Real When Needed**: 2 tests use real LLM to verify actual integration
3. **Economical Design**: Real LLM tests are designed to minimize token usage:
   - Small data (2-3 rows)
   - Simple validators (quick pass)
   - max_retries=1 (limit attempts)
   - gpt-4o-mini (cheapest model)

## Test Results

Last run: **39 passed, 17 skipped** (doctests marked with +SKIP)

- All 17 unit tests passed (15 mock + 2 real LLM)
- All 22 runnable doctests passed
- 17 example doctests skipped (illustrative only)
