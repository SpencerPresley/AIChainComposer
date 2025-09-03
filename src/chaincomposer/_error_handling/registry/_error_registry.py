from __future__ import annotations

from typing import Type, Dict, TypeVar, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..classes import _ChainError

class _ErrorRegistry:
    """Registry for mapping exception types to their handlers."""
    
    _registry: Dict[Type[Exception], Type[_ChainError]] = {}
    
    @classmethod
    def register(
        cls,
        handled_exception: Type[Exception],
    ) -> Callable[[Type[_ChainError]], Type[_ChainError]]:
        """Decorator to register an error handler for a specific exception type.
        
        Args:
            handled_exception: The exception type this handler can process
            
        Returns:
            Callable[[Type[T]], Type[T]]: A decorator that registers the handler for the exception type.
        """
        def decorator(handler_cls: Type[_ChainError]) -> Type[_ChainError]:
            cls._registry[handled_exception] = handler_cls
            return handler_cls
        return decorator
    
    @classmethod
    def get_handler(
        cls,
        exception: Exception,
    ) -> Type[_ChainError] | None:
        """Get the appropriate handler for an exception type.
        
        Args:
            exception: The exception to get a handler for.
            
        Returns:
            The handler for the exception type, or None if no handler is found.
        """
        for exc_type, handler in cls._registry.items():
            if isinstance(exception, exc_type):
                return handler
        return None