from typing import Union

from ..registry import _ErrorRegistry
from ..enums import _BaseErrorReference
from ..classes import _ChainError

class _ChainExceptionFactory:
    """Factory class for creating chain-specific exceptions."""
    
    @staticmethod
    def create_error(
        error: Exception,
        error_reference: _BaseErrorReference | None = None,
        message: str | None = None,
    ) -> _ChainError:
        """Creates appropriate chain exception based on the error type.

        Args:
            error (Exception): The original error that occurred
            error_reference (_BaseErrorReference | None): Reference for looking up error messages
            message (str | None): Optional custom error message
            
        Returns:
            _ChainError: The appropriate chain exception type
        """
        handler_cls = _ErrorRegistry.get_handler(error)
        
        if handler_cls is not None:
            return handler_cls(
                original_error=error,
                error_reference=error_reference
            )
            
        return _ChainError(
            message=message,
            original_error=error,
            error_reference=error_reference
        )
