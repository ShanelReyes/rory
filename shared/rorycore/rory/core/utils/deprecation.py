import functools
import warnings


def deprecated(replacement: str):
    """Decorator to mark functions or methods as deprecated.

    Emits a DeprecationWarning on each call with a hint pointing
    to the replacement.

    Args:
        replacement: Name of the replacement function/method.

    Returns:
        A decorator that emits DeprecationWarning then delegates.
    """
    def decorator(func):
        """Return a wrapped version of func that emits a DeprecationWarning.

        Args:
            func: The original function to wrap.

        Returns:
            callable: Wrapped function with deprecation behaviour.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Emit a DeprecationWarning and delegate to the original function.

            Args:
                *args: Positional arguments passed to the original function.
                **kwargs: Keyword arguments passed to the original function.

            Returns:
                The return value of the original function.
            """
            warnings.warn(
                f"{func.__qualname__} is deprecated. "
                f"Use {replacement} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator
