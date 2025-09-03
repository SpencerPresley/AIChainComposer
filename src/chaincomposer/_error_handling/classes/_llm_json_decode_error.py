from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ._base_chain_error import _ChainError
from ..utils import get_error_type, _ERROR_MESSAGE_MAP
from ..enums import _BaseErrorReference, _ErrorType
from ..registry import _ErrorRegistry

if TYPE_CHECKING:
    from ..enums import _BaseErrorReference

@_ErrorRegistry.register(json.JSONDecodeError)
class _LLMJsonDecodeError(_ChainError):
    """Exception for JSON parsing failures in LLM output."""
    
    def __init__(
        self, 
        original_error: json.JSONDecodeError,
        error_reference: _BaseErrorReference | None = None,
    ) -> None:
        """Initialize the LLMJsonDecodeError.
        
        Args:
            original_error (json.JSONDecodeError): The original JSON decode error
            error_reference (BaseErrorReference | None): The error reference that occurred.
        """
        super().__init__(
            original_error=original_error,
            error_reference=error_reference,
            class_error_type=get_error_type(original_error),
        )