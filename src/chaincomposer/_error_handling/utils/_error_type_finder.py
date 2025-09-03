import json
from pydantic import ValidationError
from ..enums._error_types import _ErrorType

def get_error_type(error: Exception) -> _ErrorType:
    """Get the error type from an exception.

    Args:
        error (Exception): The exception to get the error type from.

    Returns:
        _ErrorType: The error type.
    """
    if isinstance(error, json.JSONDecodeError):
        return _ErrorType.JSON_ERROR
    elif isinstance(error, ValidationError):
        return _ErrorType.VALIDATION_ERROR
    elif isinstance(error, ValueError):
        return _ErrorType.VALUE_ERROR
    elif isinstance(error, TypeError):
        return _ErrorType.TYPE_ERROR
    else:
        return _ErrorType.UNKNOWN_ERROR