# Tests for Causal Pool Evaluation

This directory contains comprehensive tests for the evaluation script (`eval/eval.py`).

## Running Tests

### Install test dependencies

```bash
uv sync --extra dev
# or
pip install pytest pytest-asyncio
```

### Run all tests

```bash
pytest tests/
```

### Run with verbose output

```bash
pytest tests/ -v
```

### Run specific test file

```bash
pytest tests/test_eval.py
```

### Run specific test class or function

```bash
pytest tests/test_eval.py::TestGetMetrics
pytest tests/test_eval.py::TestGetMetrics::test_exactly_correct_single_option
```

### Run with coverage

```bash
pytest tests/ --cov=eval --cov-report=html
```

## Test Coverage

The test suite covers:

1. **Model Name Normalization** (`TestNormalizeModelName`)
   - Various model name formats
   - Handling of slashes, dashes, and case

2. **Metrics Calculation** (`TestGetMetrics`)
   - Exactly correct predictions
   - Partially correct predictions
   - Wrong predictions
   - Invalid predictions (duplicates, lowercase, non-alpha)
   - Edge cases (empty, whitespace)

3. **Video Processing** (`TestGetVideoDuration`, `TestCutVideoFirstHalf`)
   - Video duration extraction using ffprobe
   - Video cutting to first half using ffmpeg
   - Error handling (missing tools, invalid files)
   - Fallback from codec copy to re-encoding

4. **Prompt Building** (`TestBuildPrompt`)
   - Descriptive questions (full video)
   - Predictive questions (first half video)
   - Unknown question types (full video)
   - Prompt formatting

5. **Async Evaluator** (`TestAsyncEvaluator`)
   - Initialization
   - Single entry evaluation
   - Multiple samples per entry

6. **Dataset Evaluation** (`TestEvaluateDataset`)
   - Question type tracking
   - Per-question-type metrics calculation
   - @1 and @k metrics per question type
   - Error handling with question types

## Test Structure

Tests use:
- **pytest** as the testing framework
- **pytest-asyncio** for async test support
- **unittest.mock** for mocking external dependencies (ffmpeg, API calls)

## Notes

- Tests mock external dependencies (ffmpeg, OpenAI API) to avoid requiring actual tools/services
- Video processing tests use temporary files and directories
- Async tests are properly marked with `@pytest.mark.asyncio`
- Tests are designed to run quickly without requiring actual video files or API access

