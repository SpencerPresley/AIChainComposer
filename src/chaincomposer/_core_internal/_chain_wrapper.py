from __future__ import annotations

import json
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    cast,
    TYPE_CHECKING,
)

from langchain.schema.runnable import Runnable
from pydantic import BaseModel, ValidationError

from .._logging import get_logger, WARNING
from .._error_handling import ( # type: ignore[attr-defined]
    _ChainExceptionFactory,
    _ChainWrapperErrorReference,
)

if TYPE_CHECKING:
    from .._type_aliases import (
        ParserType,
        FallbackParserType,
    )


class _ChainWrapper:
    """A wrapper class for managing and executing chains with optional fallback chains.

    This class is designed to handle the execution of primary chains and, if necessary,
    fallback chains in case of errors or unexpected outputs. It supports preprocessing
    and postprocessing of input and output data, as well as logging for debugging and
    monitoring purposes.

    Attributes:
        chain (Runnable): The primary chain to be executed.
        fallback_chain (Runnable): The fallback chain to be executed if the primary chain fails.
        parser (ParserType | None): The parser to be used for processing the output of the primary chain.
            Defaults to None.
        fallback_parser (FallbackParserType | None): The parser to be used for processing the output of the fallback chain.
            Defaults to None.
        preprocessor (Callable | None): A function to preprocess input data before passing it to the chain.
            Defaults to None.
        postprocessor (Callable | None): A function to postprocess the output data from the chain.
            Defaults to None.
        level (logging.Level | None): The log level to use for the module.
            Defaults to logging.WARNING.
        debug (bool | None): Whether to enable debug logging.
            Defaults to False.
    """

    def __init__(
        self,
        *,
        chain: Runnable,
        fallback_chain: Runnable,
        parser: ParserType | None = None,
        fallback_parser: FallbackParserType | None = None,
        preprocessor: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
        postprocessor: Callable[[Any], Any] | None = None,
        enable_logging: bool | None = False,
        level: int | None = WARNING,
        debug: bool | None = False,
    ) -> None:
        """Initialize the ChainBuilder.

        Args:
            chain (Runnable): The primary chain to be executed.
            fallback_chain (Runnable): The fallback chain to be executed if the primary chain fails.
            parser (ParserType | None): The parser to be used for the primary chain.
            fallback_parser (FallbackParserType | None): The parser to be used for the fallback chain.
            preprocessor (Callable | None): A function to preprocess input data before passing it to the chain.
                Defaults to None.
            postprocessor (Callable | None): A function to postprocess the output data from the chain.
                Defaults to None.
            level (logging.Level | None): The log level to use for the module.
                Defaults to logging.WARNING.
            debug (bool | None): Whether to enable debug logging.
                Defaults to False.
        """
        self.logger = get_logger(
            module_name=__name__,
            level=level,
            null_logger=not enable_logging,
        )

        self.parser: Optional[ParserType] = parser
        self.fallback_parser: Optional[FallbackParserType] = fallback_parser
        self.preprocessor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = (
            preprocessor
        )

        self.chain: Runnable = chain
        self.fallback_chain: Runnable = fallback_chain
        self.postprocessor: Optional[Callable[[Any], Any]] = postprocessor
        if debug is None:
            warnings.warn(
                "The `debug` argument takes an optional boolean value, you gave None. "
                "The value will be set to False.\n"
                "Additionally, if your intention is to not enable debug logging, "
                "you can simply omit the `debug` argument.\n"
            )
        self.debug: bool = debug if debug is not None else False

    def __str__(self) -> str:
        """Returns a string representation of the ChainWrapper object.

        The string includes the chain, the type of the parser, and the type of the fallback parser.

        Returns:
            str: A string representation of the ChainWrapper object.
        """
        return f"ChainWrapper(chain={self.chain}, parser={type(self.parser).__name__ if self.parser else 'None'}, fallback_parser={type(self.fallback_parser).__name__ if self.fallback_parser else 'None'})"

    def __repr__(self) -> str:
        """Returns a detailed string representation for debugging.
        
        Returns:
            str: A detailed string representation of the ChainWrapper object.
        """
        return (
            f"ChainWrapper(\n"
            f"    chain={self.chain!r},\n"
            f"    parser={self.parser!r},\n"
            f"    fallback_parser={self.fallback_parser!r},\n"
            f"    preprocessor={self.preprocessor!r},\n"
            f"    postprocessor={self.postprocessor!r}\n"
            f")"
        )

    def run_chain(
        self,
        *,
        input_data: Dict[str, Any] | None = None,
        is_last_chain: bool | None = False,
    ) -> Any:
        """Executes the primary chain with the provided input data.

        If an error occurs in the primary chain, and a fallback chain is provided,
        it attempts to execute the fallback chain.

        Args:
            input_data (dict | None): The input data required to run the chain.
                Type: Dict[str, Any]
                Defaults to None.
            is_last_chain (bool | None): Indicates if this is the last chain in the sequence.
                Defaults to False.

        Returns:
            Any: The output from the chain execution, potentially processed by a postprocessor.

        Raises:
            ValueError: If no input data is provided.
            json.JSONDecodeError: If a JSON decode error occurs in both the main and fallback chains.
            ValidationError: If a validation error occurs in both the main and fallback chains.
            TypeError: If a type error occurs in both the main and fallback chains.
        """
        if is_last_chain is None:
            warnings.warn(
                "The `is_last_chain` argument takes an optional boolean value, you gave None. "
                "The value will be set to False.\n"
                "Additionally, if your intention is to not enable debug logging, "
                "you can simply omit the `debug` argument.\n"
            )
        is_last_chain: bool = ( # type: ignore[no-redef]
            False 
            if (is_last_chain is None) or (not isinstance(is_last_chain, bool))
            else is_last_chain 
        ) 
        assert isinstance(is_last_chain, bool)
        
        if (input_data is not None) and (self.preprocessor is not None):
            input_data = self.preprocessor(input_data) # type: ignore[no-redef]

        try:
            # Attempt to invoke the primary chain
            output: Any = self.chain.invoke(input_data)
        except (
            json.JSONDecodeError,
            ValidationError,
            ValueError,
            TypeError,
        ) as main_chain_exception:
            if not self._should_run_fallback_chain(is_last_chain):
                self._handle_main_chain_error(
                    main_chain_exception, 
                    is_last_chain
                )
            else:
                try:
                    # Attempt to invoke the fallback chain
                    output: Any = self.fallback_chain.invoke(input_data) # type: ignore[no-redef]
                except (
                    json.JSONDecodeError,
                    ValidationError,
                    ValueError,
                    TypeError,
                ) as fallback_exception:
                    raise _ChainExceptionFactory.create_error(
                        error=fallback_exception,
                        error_reference=_ChainWrapperErrorReference.FALLBACK_CHAIN_ERROR
                    ) from fallback_exception

        # Only need to handle intermediate chain Pydantic outputs
        # Rest are handled by JsonOutputParser or PydanticOutputParser
        if not is_last_chain and isinstance(output, BaseModel):
            return output.model_dump()

        if self.postprocessor:
            output: Any = self.postprocessor(output) # type: ignore[no-redef]

        return output

    def get_parser_type(self) -> str | None:
        """Returns the type of the parser as a string if the parser exists, otherwise returns None.

        Returns:
            str | None: The type of the parser as a string, or None if the parser
                does not exist.
        """
        return type(self.parser).__name__ if self.parser else None

    def get_fallback_parser_type(self) -> str | None:
        """Returns the type name of the fallback parser if it exists, otherwise returns None.

        Returns:
            str | None: The type name of the fallback parser as a string if it exists,
                otherwise None.
        """
        return type(self.fallback_parser).__name__ if self.fallback_parser else None
    
    def _should_run_fallback_chain(self, is_last_chain: bool) -> bool:
        """Determines if the fallback chain should be run.

        Args:
            is_last_chain (bool): Indicates if this is the last chain in the sequence.
                Defaults to False.
        """
        return (self.fallback_chain is not None) and (not is_last_chain)

    def _handle_main_chain_error(
        self,
        main_chain_exception: Exception,
        is_last_chain: bool,
    ) -> None:
        """Handles the main chain error.

        Args:
            main_chain_exception (Exception): The exception that occurred in the main chain.
            is_last_chain (bool): Indicates if this is the last chain in the sequence.
        
        Returns:
            None: if the fallback chain is provided and it is not the last chain.

        Raises:
            Exception: The exception that occurred in the main chain if there is no fallback chain or it is the last chain.
        """
        if self.fallback_chain and not is_last_chain:
            self.logger.error(
                f"Error in main chain, attempting to execute fallback chain."
            )
        elif is_last_chain:
            self.logger.error(
                f"Error in main chain and it is the last chain, raising error."
            )
            raise _ChainExceptionFactory.create_error(
                error=main_chain_exception,
                error_reference=_ChainWrapperErrorReference.MAIN_CHAIN_ERROR
            )
        else:
            self.logger.error(
                f"Error in main chain and it is not the last chain, no fallback chain provided. Raising error."
            )
            raise _ChainExceptionFactory.create_error(
                error=main_chain_exception,
                error_reference=_ChainWrapperErrorReference.MAIN_CHAIN_ERROR
            )
