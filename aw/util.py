"""Utility tools and helper functions for agents.

Includes file sampling, LLM facades, and other common utilities.
"""

from typing import Any, Union, Callable
from pathlib import Path
from functools import partial
import urllib.parse
import io
import os

from config2py import get_app_data_folder, process_path

# ============================================================================
# Persistent Storage Configuration
# ============================================================================

# Get AW data directory from env var or use default app data folder
# Users can override by setting AW_DATA_DIR environment variable
# Default location follows XDG standards:
#   - Linux/macOS: ~/.local/share/aw
#   - Windows: %LOCALAPPDATA%/aw
_aw_data_dir_raw = os.environ.get('AW_DATA_DIR') or get_app_data_folder('aw')

# Process path and ensure it exists
AW_DATA_DIR = process_path(_aw_data_dir_raw, ensure_dir_exists=True)

# Convenience function to join paths relative to AW_DATA_DIR
# Example: djoin('models', 'model1.pkl') -> '/Users/user/.local/share/aw/models/model1.pkl'
djoin = partial(os.path.join, AW_DATA_DIR)
djoin.rootdir = AW_DATA_DIR


class FileSamplerTool:
    """Tool to sample and analyze file metadata and content.

    Helps agents understand what kind of data they're dealing with
    before attempting to load it.

    Example:
        >>> sampler = FileSamplerTool()  # doctest: +SKIP
        >>> info = sampler('/path/to/data.csv')  # doctest: +SKIP
        >>> info['extension']  # doctest: +SKIP
        '.csv'
    """

    def __init__(self, sample_size: int = 1024):
        """Initialize file sampler.

        Args:
            sample_size: Number of bytes to sample from file
        """
        self.sample_size = sample_size

    def __call__(self, uri: str) -> dict[str, Any]:
        """Sample file and return metadata.

        Args:
            uri: File URI (local path or URL)

        Returns:
            Dict with extension, sample_bytes, encoding, etc.
        """
        parsed = urllib.parse.urlparse(uri)

        # Determine if local or remote
        if parsed.scheme in ('', 'file'):
            return self._sample_local(parsed.path or uri)
        elif parsed.scheme in ('http', 'https'):
            return self._sample_remote(uri)
        else:
            return {'error': f'Unsupported URI scheme: {parsed.scheme}'}

    def _sample_local(self, path: str) -> dict[str, Any]:
        """Sample local file."""
        file_path = Path(path)

        if not file_path.exists():
            return {'error': f'File not found: {path}'}

        info = {
            'uri': str(file_path),
            'extension': file_path.suffix,
            'size': file_path.stat().st_size,
            'exists': True,
        }

        try:
            # Read sample
            with open(file_path, 'rb') as f:
                sample_bytes = f.read(self.sample_size)
            info['sample_bytes'] = sample_bytes

            # Try to decode as text
            try:
                info['sample_text'] = sample_bytes.decode('utf-8')
                info['encoding'] = 'utf-8'
            except UnicodeDecodeError:
                info['encoding'] = 'binary'
        except Exception as e:
            info['error'] = str(e)

        return info

    def _sample_remote(self, url: str) -> dict[str, Any]:
        """Sample remote file via HTTP."""
        try:
            import urllib.request

            # Get headers
            with urllib.request.urlopen(url) as response:
                info = {
                    'uri': url,
                    'content_type': response.headers.get('Content-Type'),
                    'content_length': response.headers.get('Content-Length'),
                }

                # Infer extension from URL or content-type
                parsed = urllib.parse.urlparse(url)
                path = parsed.path
                if '.' in path:
                    info['extension'] = '.' + path.split('.')[-1]

                # Read sample
                sample_bytes = response.read(self.sample_size)
                info['sample_bytes'] = sample_bytes

                try:
                    info['sample_text'] = sample_bytes.decode('utf-8')
                    info['encoding'] = 'utf-8'
                except UnicodeDecodeError:
                    info['encoding'] = 'binary'

            return info
        except Exception as e:
            return {'uri': url, 'error': str(e)}


def infer_loader_from_extension(extension: str) -> str:
    """Infer pandas loader function from file extension.

    Args:
        extension: File extension (e.g., '.csv', '.json')

    Returns:
        Name of pandas loader function

    Example:
        >>> infer_loader_from_extension('.csv')
        'read_csv'
        >>> infer_loader_from_extension('.xlsx')
        'read_excel'
    """
    ext_map = {
        '.csv': 'read_csv',
        '.tsv': 'read_csv',
        '.txt': 'read_csv',
        '.json': 'read_json',
        '.jsonl': 'read_json',
        '.xlsx': 'read_excel',
        '.xls': 'read_excel',
        '.parquet': 'read_parquet',
        '.feather': 'read_feather',
        '.hdf': 'read_hdf',
        '.h5': 'read_hdf',
        '.sql': 'read_sql',
        '.html': 'read_html',
        '.xml': 'read_xml',
        '.pickle': 'read_pickle',
        '.pkl': 'read_pickle',
    }

    return ext_map.get(extension.lower(), 'read_csv')  # Default to CSV


def infer_loader_params(extension: str, sample_text: str = None) -> dict:
    """Infer parameters for pandas loader based on file characteristics.

    Args:
        extension: File extension
        sample_text: Optional sample of file content

    Returns:
        Dict of parameters to pass to loader

    Example:
        >>> params = infer_loader_params('.csv', 'a,b,c\\n1,2,3')  # doctest: +SKIP
        >>> params.get('sep')  # doctest: +SKIP
        ','
    """
    params = {}

    # Extension-based defaults
    if extension == '.tsv':
        params['sep'] = '\t'
    elif extension == '.jsonl':
        params['lines'] = True

    # Sample-based inference
    if sample_text:
        # Check for common delimiters
        if '\t' in sample_text and extension in ('.txt', '.csv'):
            params['sep'] = '\t'
        elif ';' in sample_text and ',' not in sample_text:
            params['sep'] = ';'
        elif '|' in sample_text:
            params['sep'] = '|'

    return params


# ============================================================================
# LLM Facades
# ============================================================================


def create_openai_chat(model: str = "gpt-4", **default_kwargs) -> Callable[[str], str]:
    """Create a text-to-text chat function using OpenAI API.

    Args:
        model: OpenAI model name
        **default_kwargs: Default parameters for API calls

    Returns:
        A callable that takes a prompt and returns a response

    Example:
        >>> chat = create_openai_chat("gpt-4")
        >>> response = chat("What is 2+2?")
    """
    try:
        from openai import OpenAI

        client = OpenAI()

        def chat(prompt: str, **kwargs) -> str:
            merged_kwargs = {**default_kwargs, **kwargs}
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **merged_kwargs,
            )
            return response.choices[0].message.content

        return chat
    except ImportError:
        raise ImportError(
            "OpenAI package not available. Install with: pip install openai"
        )


def create_oa_chat() -> Callable[[str], str]:
    """Create a chat function using the oa package.

    Returns:
        A callable that takes a prompt and returns a response

    Example:
        >>> chat = create_oa_chat()
        >>> response = chat("Hello!")
    """
    try:
        from oa import chat as oa_chat

        return oa_chat
    except ImportError:
        raise ImportError("oa package not available. Install with: pip install oa")


def default_llm_factory(model: Union[str, Callable] = "gpt-4") -> Callable[[str], str]:
    """Factory to create LLM function from various specifications.

    Args:
        model: Either a model name string, or a callable

    Returns:
        A callable text-to-text function

    Example:
        >>> llm = default_llm_factory("gpt-4")
        >>> llm = default_llm_factory(lambda p: f"Echo: {p}")
    """
    if callable(model):
        return model

    # Try to use oa package first (as specified in requirements)
    try:
        return create_oa_chat()
    except ImportError:
        pass

    # Fall back to OpenAI
    try:
        return create_openai_chat(model)
    except ImportError:
        pass

    # Ultimate fallback - echo function
    def echo_llm(prompt: str) -> str:
        return f"[ECHO MODE - No LLM available] Received: {prompt[:100]}"

    return echo_llm


# ============================================================================
# DataFrame Utilities
# ============================================================================


def compute_dataframe_info(df) -> dict:
    """Compute comprehensive info about a DataFrame.

    Args:
        df: pandas DataFrame

    Returns:
        Dict with shape, dtypes, null counts, sample rows, etc.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> info = compute_dataframe_info(df)
        >>> info['shape']
        (2, 2)
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        return {'error': 'Not a DataFrame', 'type': type(df).__name__}

    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'null_counts': df.isnull().sum().to_dict(),
        'total_nulls': int(df.isnull().sum().sum()),
        'memory_usage': int(df.memory_usage(deep=True).sum()),
    }

    # Add numeric column info
    numeric_cols = df.select_dtypes(include='number').columns
    info['numeric_columns'] = list(numeric_cols)
    info['num_numeric_columns'] = len(numeric_cols)

    # Add categorical column info
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    info['categorical_columns'] = list(categorical_cols)

    # Sample rows (as records for JSON serializability)
    info['sample_rows'] = df.head(5).to_dict('records')

    return info


def get_numeric_columns(df, exclude_nulls: bool = True):
    """Get numeric columns from DataFrame.

    Args:
        df: pandas DataFrame
        exclude_nulls: Whether to exclude columns with null values

    Returns:
        Generator of column names

    Example:
        >>> for col in get_numeric_columns(df):  # doctest: +SKIP
        ...     print(col)
    """
    import pandas as pd

    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        if exclude_nulls and df[col].isnull().any():
            continue
        yield col
