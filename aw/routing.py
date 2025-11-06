"""
Configurable routing and decision chains for agentic workflows.

This module provides tools for building extensible, inspectable routing chains
that follow the open-closed principle. Users can:
- Use default configurations out of the box
- Inspect all routing rules and mappings
- Customize individual components
- Extend with new routing strategies

The key pattern is: provide sensible defaults, make them visible, keep them mutable.
"""

from typing import Optional, Callable, Any, Protocol
from dataclasses import dataclass, field
from functools import partial
from collections.abc import Mapping


class RoutingStrategy(Protocol):
    """Protocol for routing strategies.

    A routing strategy takes a context and returns a result or None.
    """

    def __call__(self, context: Any) -> Optional[Any]:
        """Apply routing logic to context."""
        ...


@dataclass
class PriorityRouter:
    """Route through strategies in priority order, short-circuiting on first match.

    This is a simpler, more specialized version of i2.routing_forest that's
    optimized for the common case of "try these in order, stop at first success".

    >>> def try_a(x): return 'A' if x > 10 else None
    >>> def try_b(x): return 'B' if x > 5 else None
    >>> def try_c(x): return 'C'
    >>>
    >>> router = PriorityRouter([try_a, try_b, try_c])
    >>> router(15)
    'A'
    >>> router(8)
    'B'
    >>> router(3)
    'C'
    """

    strategies: list[RoutingStrategy] = field(default_factory=list)

    def __call__(self, context: Any) -> Optional[Any]:
        """Route through strategies in order, return first non-None result."""
        for strategy in self.strategies:
            result = strategy(context)
            if result is not None:
                return result
        return None

    def prepend(self, strategy: RoutingStrategy) -> 'PriorityRouter':
        """Add a strategy at the beginning (highest priority)."""
        return PriorityRouter([strategy] + self.strategies)

    def append(self, strategy: RoutingStrategy) -> 'PriorityRouter':
        """Add a strategy at the end (lowest priority)."""
        return PriorityRouter(self.strategies + [strategy])

    def insert(self, index: int, strategy: RoutingStrategy) -> 'PriorityRouter':
        """Insert a strategy at a specific position."""
        new_strategies = self.strategies.copy()
        new_strategies.insert(index, strategy)
        return PriorityRouter(new_strategies)


@dataclass
class ConditionalRouter(PriorityRouter):
    """Router with short-circuit conditions.

    Only accepts results that match the short-circuit condition,
    otherwise continues to next strategy.

    >>> def try_url(x): return x.get('url_ext')
    >>> def try_type(x): return x.get('content_type_ext')
    >>>
    >>> # Only accept .pdf or .md, otherwise keep trying
    >>> router = ConditionalRouter(
    ...     [try_url, try_type],
    ...     short_circuit=lambda r: r in {'.pdf', '.md'}
    ... )
    >>>
    >>> # .pdf matches short-circuit, return immediately
    >>> router({'url_ext': '.pdf', 'content_type_ext': '.html'})
    '.pdf'
    >>>
    >>> # .jpg doesn't match, try next strategy, .md matches
    >>> router({'url_ext': '.jpg', 'content_type_ext': '.md'})
    '.md'
    """

    short_circuit: Optional[Callable[[Any], bool]] = None

    def __call__(self, context: Any) -> Optional[Any]:
        """Route with short-circuit logic.

        If short_circuit is defined:
        - Results matching the condition are returned immediately
        - Results not matching are ignored, and next strategy is tried

        If no short_circuit is defined, behaves like PriorityRouter.
        """
        for strategy in self.strategies:
            result = strategy(context)
            if result is not None:
                # If no short-circuit function, return first result (PriorityRouter behavior)
                if self.short_circuit is None:
                    return result
                # If short-circuit matches, return immediately
                if self.short_circuit(result):
                    return result
                # Otherwise, ignore this result and try next strategy

        return None


@dataclass
class MappingRouter:
    """Router based on a lookup mapping.

    Provides a clean interface for switch-case style routing with:
    - Visible, mutable mapping
    - Optional default value or factory
    - Easy extension via dict interface

    >>> # Content-Type to extension mapping
    >>> ct_map = {
    ...     'application/pdf': '.pdf',
    ...     'text/html': '.html',
    ...     'application/json': '.json',
    ... }
    >>> router = MappingRouter(ct_map)
    >>>
    >>> router('application/pdf')
    '.pdf'
    >>> router('text/plain') is None  # No default
    True
    >>>
    >>> # With default
    >>> router_with_default = MappingRouter(ct_map, default='.bin')
    >>> router_with_default('text/plain')
    '.bin'
    """

    mapping: dict = field(default_factory=dict)
    default: Any = None
    key_transform: Optional[Callable[[Any], Any]] = None

    def __call__(self, key: Any) -> Any:
        """Look up key in mapping."""
        if self.key_transform:
            key = self.key_transform(key)
        return self.mapping.get(key, self.default)

    def __getitem__(self, key):
        """Access mapping directly."""
        return self.mapping[key]

    def __setitem__(self, key, value):
        """Update mapping directly."""
        self.mapping[key] = value

    def update(self, *args, **kwargs):
        """Update mapping."""
        self.mapping.update(*args, **kwargs)

    def copy(self) -> 'MappingRouter':
        """Create a copy with same configuration."""
        return MappingRouter(
            self.mapping.copy(), default=self.default, key_transform=self.key_transform
        )


# ---------------------------------------------------------------------------
# Extension Detection - Concrete Application
# ---------------------------------------------------------------------------


@dataclass
class ExtensionContext:
    """Context for extension detection.

    >>> ctx = ExtensionContext(
    ...     url='https://example.com/file.pdf',
    ...     content=b'%PDF-1.5',
    ...     content_type='text/html'
    ... )
    >>> ctx.url
    'https://example.com/file.pdf'
    """

    url: str
    content: bytes = b''
    content_type: str = ''
    explicit_extension: Optional[str] = None


# Default Content-Type mapping (visible and mutable)
DEFAULT_CONTENT_TYPE_MAP = {
    'application/pdf': '.pdf',
    'text/html': '.html',
    'text/markdown': '.md',
    'text/plain': '.txt',
    'application/json': '.json',
    'application/xml': '.xml',
    'text/csv': '.csv',
    'image/png': '.png',
    'image/jpeg': '.jpg',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/svg+xml': '.svg',
    'application/zip': '.zip',
    'application/gzip': '.gz',
    'application/x-tar': '.tar',
}


# Default magic bytes patterns (visible and mutable)
DEFAULT_MAGIC_BYTES_MAP = {
    b'%PDF': '.pdf',
    b'<!DOCTYPE html': '.html',
    b'<!doctype html': '.html',
    b'<html': '.html',
    b'\x89PNG': '.png',
    b'\xff\xd8\xff': '.jpg',
    b'GIF87a': '.gif',
    b'GIF89a': '.gif',
    b'PK\x03\x04': '.zip',
    b'\x1f\x8b': '.gz',
}


def detect_from_url(ctx: ExtensionContext) -> Optional[str]:
    """Extract extension from URL path.

    >>> ctx = ExtensionContext('https://example.com/file.pdf')
    >>> detect_from_url(ctx)
    '.pdf'
    >>> ctx = ExtensionContext('https://example.com/file')
    >>> detect_from_url(ctx) is None
    True
    """
    from urllib.parse import urlparse
    from pathlib import Path

    url_path = urlparse(ctx.url).path
    ext = Path(url_path).suffix.lower()

    # Validate extension looks reasonable
    if ext and 1 < len(ext) <= 5 and ext[1:].replace('_', '').isalnum():
        return ext

    return None


def make_content_type_detector(
    content_type_map: Optional[dict] = None,
) -> Callable[[ExtensionContext], Optional[str]]:
    """Factory for content-type based detection.

    Returns a detector function with the given mapping.
    Users can create custom detectors with different mappings.

    >>> # Use default mapping
    >>> detector = make_content_type_detector()
    >>> ctx = ExtensionContext('', content_type='application/pdf')
    >>> detector(ctx)
    '.pdf'
    >>>
    >>> # Custom mapping
    >>> custom_map = {'text/plain': '.log'}
    >>> custom_detector = make_content_type_detector(custom_map)
    >>> ctx = ExtensionContext('', content_type='text/plain')
    >>> custom_detector(ctx)
    '.log'
    """
    if content_type_map is None:
        content_type_map = DEFAULT_CONTENT_TYPE_MAP.copy()

    def detect(ctx: ExtensionContext) -> Optional[str]:
        if not ctx.content_type:
            return None

        # Clean content type (remove charset, etc.)
        content_type = ctx.content_type.split(';')[0].strip().lower()
        return content_type_map.get(content_type)

    return detect


def make_magic_bytes_detector(
    magic_bytes_map: Optional[dict] = None,
) -> Callable[[ExtensionContext], Optional[str]]:
    """Factory for magic bytes based detection.

    >>> detector = make_magic_bytes_detector()
    >>> ctx = ExtensionContext('', content=b'%PDF-1.5...')
    >>> detector(ctx)
    '.pdf'
    >>> ctx = ExtensionContext('', content=b'<html>...')
    >>> detector(ctx)
    '.html'
    """
    if magic_bytes_map is None:
        magic_bytes_map = DEFAULT_MAGIC_BYTES_MAP.copy()

    def detect(ctx: ExtensionContext) -> Optional[str]:
        if not ctx.content or len(ctx.content) < 4:
            return None

        # Check against magic byte patterns
        content_lower = ctx.content[:100].lower()
        for magic, ext in magic_bytes_map.items():
            if content_lower.startswith(magic.lower()):
                return ext

        # Try imghdr for images
        try:
            import imghdr

            img_ext = imghdr.what(None, h=ctx.content[:50])
            if img_ext:
                return f'.{img_ext}'
        except Exception:
            pass

        return None

    return detect


def detect_explicit(ctx: ExtensionContext) -> Optional[str]:
    """Use explicit extension if provided.

    >>> ctx = ExtensionContext('', explicit_extension='pdf')
    >>> detect_explicit(ctx)
    '.pdf'
    >>> ctx = ExtensionContext('', explicit_extension='.json')
    >>> detect_explicit(ctx)
    '.json'
    """
    if ctx.explicit_extension:
        ext = ctx.explicit_extension
        if not ext.startswith('.'):
            ext = f'.{ext}'
        return ext
    return None


@dataclass
class ExtensionRouter:
    """Route extension detection through configurable strategies.

    This router implements the default logic for detecting file extensions,
    but every component is visible and customizable.

    Examples:
        >>> # Use with defaults
        >>> router = ExtensionRouter()
        >>> ctx = ExtensionContext('https://example.com/file.pdf', b'%PDF-1.5')
        >>> router(ctx)
        '.pdf'
        >>>
        >>> # URL without extension, falls back to magic bytes
        >>> ctx = ExtensionContext('https://example.com/download', b'%PDF-1.5')
        >>> router(ctx)
        '.pdf'
        >>>
        >>> # Priority extensions short-circuit
        >>> ctx = ExtensionContext('https://example.com/file.pdf', b'<html>')
        >>> router(ctx)  # .pdf from URL wins even though content is HTML
        '.pdf'
        >>>
        >>> # Customize content type mapping
        >>> router.content_type_map['.txt'] = 'text/x-log'
        >>>
        >>> # Add custom detector
        >>> def my_detector(ctx): return '.custom' if 'special' in ctx.url else None
        >>> router = router.with_prepended_strategy(my_detector)
    """

    # Priority extensions that short-circuit other checks
    priority_extensions: frozenset = field(
        default_factory=lambda: frozenset(['.pdf', '.md', '.json'])
    )

    # Configurable mappings (users can modify these)
    content_type_map: dict = field(
        default_factory=lambda: DEFAULT_CONTENT_TYPE_MAP.copy()
    )
    magic_bytes_map: dict = field(
        default_factory=lambda: DEFAULT_MAGIC_BYTES_MAP.copy()
    )

    # Detection strategies (built in __post_init__)
    router: Optional[ConditionalRouter] = field(default=None, init=False)

    def __post_init__(self):
        """Build the routing chain."""
        # Create detectors with current mappings
        content_type_detector = make_content_type_detector(self.content_type_map)
        magic_bytes_detector = make_magic_bytes_detector(self.magic_bytes_map)

        # Build strategy chain: explicit → URL → content-type → magic bytes
        strategies = [
            detect_explicit,
            detect_from_url,
            content_type_detector,
            magic_bytes_detector,
        ]

        # Short-circuit on priority extensions
        self.router = ConditionalRouter(
            strategies, short_circuit=lambda ext: ext in self.priority_extensions
        )

    def __call__(
        self,
        url: str = '',
        content: bytes = b'',
        content_type: str = '',
        *,
        explicit_extension: Optional[str] = None,
    ) -> str:
        """Detect extension from context.

        Can be called with ExtensionContext or individual args.
        """
        # Support both context object and individual args
        if isinstance(url, ExtensionContext):
            ctx = url
        else:
            ctx = ExtensionContext(url, content, content_type, explicit_extension)

        result = self.router(ctx)
        return result if result else '.bin'

    def with_prepended_strategy(
        self, strategy: Callable[[ExtensionContext], Optional[str]]
    ) -> 'ExtensionRouter':
        """Create new router with added strategy at highest priority."""
        new_router = ExtensionRouter(
            self.priority_extensions,
            self.content_type_map.copy(),
            self.magic_bytes_map.copy(),
        )
        new_router.router = self.router.prepend(strategy)
        return new_router

    def with_appended_strategy(
        self, strategy: Callable[[ExtensionContext], Optional[str]]
    ) -> 'ExtensionRouter':
        """Create new router with added strategy at lowest priority."""
        new_router = ExtensionRouter(
            self.priority_extensions,
            self.content_type_map.copy(),
            self.magic_bytes_map.copy(),
        )
        new_router.router = self.router.append(strategy)
        return new_router


# Default instance for convenience
extension_router = ExtensionRouter()
