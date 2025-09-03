import logging
from typing import Dict

class ColorFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels

    Attributes:
        COLOR_MAP (Dict[str, str]): A dictionary mapping log levels to their corresponding colors.

    Methods:
        Public Methods:
            format: Format the log record with colors.
    """

    COLOR_MAP: Dict[str, str] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
        "RESET": "\033[0m",  # Reset color
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log record with colors.
        """
        # Color only for console output (StreamHandler)
        handler = logging.getLogger(record.name).handlers[0]
        if type(handler) is logging.StreamHandler:
            levelname = record.levelname
            if levelname in self.COLOR_MAP:
                record.levelname = f"{self.COLOR_MAP[levelname]}{levelname}{self.COLOR_MAP['RESET']}"
        return super().format(record)


# Keep track of configured loggers to
# avoid re-configuring the same logger multiple times
# for seperate class instances.
#
# Implements a singleton pattern for loggers of each unique name.
#
# Results in only one logger instance per unique name (i.e. a class object)
# per runtime instance of the program.
_configured_loggers: Dict[str, logging.Logger] = {}

# Log levels for export across the package
# so that other modules can use them without
# needing to import the logging module.
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# Used to validate the log_level argument is
# a valid python logging log level
# in the get_logger() function.
VALID_LOG_LEVELS = {DEBUG, INFO, WARNING, ERROR, CRITICAL}

def _validate_log_level(level: int) -> None:
    """Validate the log level is a valid python logging log level

    Args:
        level (logging.Level): The log level to validate.

    Returns:
        None
    """
    if level not in VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log level provided: `{level}`. "
            f"Must be one of: {VALID_LOG_LEVELS}."
        )

def _is_existing_logger(module_name: str) -> bool:
    """Check if a logger already exists for the module

    Args:
        module_name (str): The name of the module to check for an existing logger.

    Returns:
        bool: True if the logger exists, False otherwise.
    """
    if module_name in _configured_loggers:
        return True
    return False

def _get_existing_logger(module_name: str) -> logging.Logger:
    """Get the existing logger for the module

    Args:
        module_name (str): The name of the module to get the logger for.

    Returns:
        logging.Logger: The existing logger for the module.
    """
    return _configured_loggers[module_name]

def _get_real_logger(
    module_name: str,
    level: int | None = WARNING,
) -> logging.Logger:
    """Configure a logger for a specific module.

    Creates a new, or returns an existing, logger for the module.
    If there is an existing logger for the module, and it is a null logger,
    a real logger will be created and returned.
    
    Args:
        module_name (str): The name of the module to configure logging for.
            This should be passed in as the `__name__` variable of the module.
        level (logging.Level | None): The log level to use for the module.
            Defaults to logging.WARNING.

    Returns:
        logging.Logger: The configured logger for the module.
    """
    if level is None:
        level = WARNING
    
    if _is_existing_logger(module_name):
        existing_logger = _get_existing_logger(module_name)
        # Check if any of the handlers is a NullHandler
        if not any(
            isinstance(handler, logging.NullHandler) 
            for handler in existing_logger.handlers
        ):
            return existing_logger
    
    # Create new logger as either:
    # - No existing logger was found
    # or
    # - Existing logger was a null logger
    logger = logging.getLogger(module_name)  
    logger.setLevel(level)
    logger.propagate = False

    # Clear any existing handlers (important for converting null logger to real logger)
    logger.handlers.clear()

    formatter = ColorFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    _configured_loggers[module_name] = logger
    return logger

def _get_null_logger(module_name: str) -> logging.Logger:
    """Get a null logger for the module

    Args:
        module_name (str): The name of the module to get the logger for.

    Returns:
        logging.Logger: The null logger for the module.
    """
    logger = logging.getLogger(module_name)
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    _configured_loggers[module_name] = logger
    return logger

def get_logger(
    module_name: str,
    level: int | None = WARNING,
    null_logger: bool | None = False,
) -> logging.Logger:
    """Get a logger for the module
    
    Validates the log level provided is a valid python logging log level.
    Returns a null logger if the null_logger argument is True,
    otherwise returns a real logger.
    
    Args:
        module_name (str): The name of the module to get the logger for.
        level (logging.Level | None): The log level to set for the logger.
            Defaults to logging.WARNING.
        null_logger (bool | None): Whether to return a null logger.
            Defaults to False.

    Returns:
        logging.Logger: The logger for the module.
    """
    if level is not None:
        _validate_log_level(level)
        
    if null_logger:
        return _get_null_logger(module_name=module_name)
    else:
        return _get_real_logger(module_name=module_name, level=level)
