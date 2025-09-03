from __future__ import annotations

from typing import TYPE_CHECKING
from ..utils import _ERROR_MESSAGE_MAP
from ..enums import _BaseErrorReference, _ErrorType

if TYPE_CHECKING:
    from ..enums import _BaseErrorReference

class _ChainError(Exception):
    """Base exception for all chain-related errors."""
    
    _CLASS_ERROR_TYPE = _ErrorType.UNKNOWN_ERROR
    _INVALID_STATE_ERROR_MESSAGE = ""
    
    def __init__(
        self,
        message: str | None = None,
        error_reference: _BaseErrorReference | None = None,
        original_error: Exception | None = None,
        class_error_type: _ErrorType | None = None,
    ) -> None:
        """Initialize the ChainError.

        Args:
            message (str): The error message.
            original_error (Exception | None, optional): The original error that occurred. Defaults to None.

        Raises:
            ValueError: If original_error is provided but is not an instance of Exception.
        """
        self.message = message
        self.original_error = original_error
        self.error_reference = error_reference
        
        if class_error_type is not None:
            self.set_class_error_type(class_error_type)
        
        if not self._is_valid_state(message, original_error, error_reference):
            raise ValueError(self._INVALID_STATE_ERROR_MESSAGE)
        
        message = message if message is not None else ""
        
        if error_reference is not None:
            self._raise_error_with_ref(original_error, error_reference)
        else:
            self._raise_error(message, original_error)
    
    def set_class_error_type(
        self, 
        error_type: _ErrorType,
    ) -> None:
        """Set the class error type.
        
        Args:
            error_type (ErrorType): The error type.
        """
        self._CLASS_ERROR_TYPE = error_type

    def _raise_error(
        self, 
        message: str, 
        original_error: Exception | None,
    ) -> None:
        """Initialize the parent Exception class with appropriate arguments.
        
        Args:
            message (str): The error message.
            original_error (Exception | None): The original error that occurred.
        """
        if self._original_error_exists(original_error):
            super().__init__(message, original_error)
        else:
            super().__init__(message)
    
    def _raise_error_with_ref(
        self,
        original_error: Exception | None,
        error_reference: _BaseErrorReference,
    ) -> None:
        """Raise the error with the error reference.
        
        Args:
            original_error (Exception | None): The original error that occurred.
            error_reference (BaseErrorReference | None): The error reference that occurred.
        """
        message = _ERROR_MESSAGE_MAP[error_reference][self._CLASS_ERROR_TYPE]
        
        if self._original_error_exists(original_error):
            super().__init__(message, original_error)
        else:
            super().__init__(message)

    def _is_valid_state(
        self, 
        message: str | None, 
        original_error: Exception | None,
        error_reference: _BaseErrorReference | None,
    ) -> bool:
        """Check if the original error is in a valid state.
        
        Args:
            message (str | None): The message that occurred.
            original_error (Exception | None): The original error that occurred.
            error_reference (BaseErrorReference | None): The error reference that occurred.

        Returns:
            bool: True if the original error is in a valid state, False otherwise
        """
        is_valid: bool = False
        message_exists: bool = self._message_exists(message)
        error_reference_exists: bool = self._error_reference_exists(error_reference)
        original_error_exists: bool = self._original_error_exists(original_error)
        original_error_is_exception: bool = self._is_exception(original_error)
        
        if original_error_exists and not original_error_is_exception:
            self._INVALID_STATE_ERROR_MESSAGE += (
                "original_error must be an instance of Exception, "
                f"but is of type: {type(original_error)}\n\n"
            )
        
        if not message_exists and not original_error_exists:
            self._INVALID_STATE_ERROR_MESSAGE += (
                "A message or original error must be provided\n\n"
            )
            
        if message_exists and error_reference_exists:
            self._INVALID_STATE_ERROR_MESSAGE += (
                "Error references are for looking up error messages, "
                "so a message should not be provided. "
                "Instead, edit your error message associated with the error reference.\n\n"
            )
        
        if message_exists or (original_error_exists and original_error_is_exception):
            is_valid = True
        
        return is_valid
        
    def _error_reference_exists(
        self, 
        error_reference: _BaseErrorReference | None
    ) -> bool:
        return error_reference is not None
    
    def _message_exists(
        self, 
        message: str | None
    ) -> bool:
        """Check if the message is None.
        
        Args:
            message (str | None): The message that occurred.

        Returns:
            bool: True if the message is None, False otherwise
        """
        return message is not None

    def _original_error_exists(
        self, 
        original_error: Exception | None
    ) -> bool:
        """Check if the original error exists.
        
        Args:
            original_error (Exception | None): The original error that occurred.

        Returns:
            bool: True if the original error exists, False otherwise
        """
        return original_error is not None
    
    def _is_exception(
        self, 
        original_error: Exception | None
    ) -> bool:
        """Check if the original error is an instance of Exception.
        
        Args:
            original_error (Exception | None): The original error that occurred.

        Returns:
            bool: True if the original error is an instance of Exception, False otherwise
 
        Note:
            This method is used to check if the original error is an instance of Exception. It is used in the `__init__` method of the `ChainError` class.
        """
        return isinstance(original_error, Exception)