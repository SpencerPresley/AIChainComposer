from __future__ import annotations

import json
from typing import TYPE_CHECKING
from pydantic import ValidationError

from ._base_chain_error import _ChainError
from ..utils import get_error_type, _ERROR_MESSAGE_MAP
from ..enums import _BaseErrorReference, _ErrorType
from ..registry import _ErrorRegistry

if TYPE_CHECKING:
    from ..enums import _BaseErrorReference

@_ErrorRegistry.register(ValidationError)
class _LLMValidationError(_ChainError):
    """Exception for Pydantic validation failures in LLM output."""
    
    def __init__(
        self, 
        original_error: ValidationError,
        error_reference: _BaseErrorReference | None = None,
    ) -> None:
        """Initialize the LLMValidationError.
        
        Args:
            original_error (ValidationError): The original Pydantic validation error
            error_reference (BaseErrorReference | None): The error reference that occurred.
        """
        super().__init__(
            original_error=original_error,
            error_reference=error_reference,
            class_error_type=get_error_type(original_error),
        )
