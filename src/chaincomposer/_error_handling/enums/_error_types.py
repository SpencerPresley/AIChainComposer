from enum import Enum

class _ErrorType(Enum):
    JSON_ERROR = "json_error"
    VALIDATION_ERROR = "validation_error"
    VALUE_ERROR = "value_error"
    TYPE_ERROR = "type_error"
    UNKNOWN_ERROR = "unknown_error"