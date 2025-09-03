from __future__ import annotations

import warnings
from typing import (
    Optional,
    Union,
    TYPE_CHECKING,
    Type,
)

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .._logging import get_logger, WARNING

if TYPE_CHECKING:
    from .._type_aliases import (
        ParserUnion,
        LLMUnion,
    )

class _ChainBuilder:
    """A builder class for constructing and configuring LangChain chains with logging and parsing capabilities.

    This class provides functionality to construct LangChain chains with customizable prompts,
    language models, and output parsers. It includes support for logging, fallback chains,
    and various parser types.

    Attributes:
        chat_prompt (ChatPromptTemplate): Template for chat prompts.
        llm (): Language model instance.

        parser (ParserType | None): Primary output parser.
        fallback_parser (ParserType | None): Fallback output parser.
        chain (:class:`Runnable`): Primary chain instance.
        fallback_chain (:class:`Runnable`): Fallback chain instance.
        debug (bool): Whether to enable debug logging.
    """

    def __init__(
        self,
        *,
        chat_prompt: ChatPromptTemplate,
        llm: LLMUnion,
        parser: ParserUnion | None = None,
        fallback_parser: ParserUnion | None = None,
        enable_logging: bool | None = False,
        level: int | None = WARNING,
        debug: bool | None = False,
    ) -> None:
        """Initialize a ChainBuilder instance.

        Args:
            chat_prompt (ChatPromptTemplate): Template for structuring chat interactions.
            llm (LLMUnion): Language model to use.
            parser (ParserType | None): Primary output parser.
                Defaults to None.
            fallback_parser (FallbackParserType | None): Fallback output parser.
                Defaults to None.
            enable_logging (bool | None): Whether to enable logging.
                Defaults to False.
            level (logging.Level | None): The log level to use for the module.
                Defaults to `logging.WARNING`.
            debug (bool | None): Whether to enable debug logging.
                Defaults to False.
        """
        self.logger = get_logger(
            module_name=__name__,
            level=level,
            null_logger=not enable_logging,
        )
        self.chat_prompt: ChatPromptTemplate = chat_prompt
        self.parser: Optional[ParserUnion] = parser
        self.fallback_parser: Optional[ParserUnion] = fallback_parser
        self.llm: Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI] = llm
        if debug is None:
            warnings.warn(
                "The `debug` argument takes an optional boolean value, you gave None. "
                "The value will be set to False.\n"
                "If your intention is to not enable debug logging, "
                "you can simply omit the `debug` argument.\n"
            )
        self.debug: bool = False if debug is None else debug
        self.chain: Runnable = self._build_chain()
        self.fallback_chain: Runnable = self._build_fallback_chain()

                
    def __str__(self) -> str:
        """Returns a concise string representation of the ChainBuilder object.
        
        Returns:
            str: A concise string representation of the ChainBuilder object.
        """
        return f"ChainBuilder(chat_prompt={type(self.chat_prompt).__name__}, llm={self.llm.__class__.__name__}, parser={type(self.parser).__name__ if self.parser else 'None'})"

    def __repr__(self) -> str:
        """Returns a detailed string representation for debugging.
        
        Returns:
            str: A detailed string representation of the ChainBuilder object.
        """
        return (
            f"ChainBuilder(\n"
            f"    chat_prompt={self.chat_prompt!r},\n"
            f"    llm={self.llm!r},\n"
            f"    parser={self.parser!r},\n"
            f"    fallback_parser={self.fallback_parser!r},\n"
            f"    chain={self.chain!r},\n"
            f"    fallback_chain={self.fallback_chain!r}\n"
            f")"
        )

    def get_chain(self) -> Runnable:
        """Retrieves the current chain.

        Returns:
            :class:`Runnable`: The current chain instance.
        """
        return self.chain

    def get_fallback_chain(self) -> Runnable:
        """Retrieve the fallback chain.

        Returns:
            :class:`Runnable`: The fallback chain instance.
        """
        return self.fallback_chain

    def _build_chain(self) -> Runnable:
        """Builds and returns a chain of `Runnable` objects.

        The chain is constructed by combining a `RunnablePassthrough` object with the `chat_prompt` and `llm` attributes. If a `parser` is provided, it is added to the chain. If `debug` is enabled, `_run_pydantic_parser_logging()` is executed.

        Returns:
            :class:`Runnable`: The constructed chain of `Runnable` objects.
        """
        # Build the base chain
        chain: Runnable = RunnablePassthrough() | self.chat_prompt | self.llm

        # Add parser if provided
        if self.parser:
            if self.debug:
                self._run_pydantic_parser_logging()
            return chain | self.parser
            
        return chain

    def _build_fallback_chain(self) -> Runnable:
        """Builds the fallback chain for the Runnable.

        This method constructs a fallback chain by combining a `RunnablePassthrough` instance with the `chat_prompt` and `llm` attributes. If a `fallback_parser` is provided, it adds the parser to the chain. If `debug` is enabled, `_run_pydantic_parser_logging()` is executed.

        Returns:
            Runnable: The constructed fallback chain.
        """
        # Build the base fallback chain
        fallback_chain: Runnable = RunnablePassthrough() | self.chat_prompt | self.llm

        # Add fallback parser if provided
        if self.fallback_parser:
            if self.debug:
                self._run_pydantic_parser_logging()
            return fallback_chain | self.fallback_parser
            
        return fallback_chain

    def _run_pydantic_parser_logging(self) -> None:
        """Performs logging related to a Pydantic parser.

        Logs the required fields and their default values (if any) for a Pydantic model if the parser is an instance of PydanticOutputParser. Additionally, it logs the JSON schema of the Pydantic model.
        """
        if isinstance(self.parser, PydanticOutputParser):
            pydantic_model: Type[BaseModel] = self.parser.pydantic_object
            self.logger.info("Required fields in Pydantic model:")
            for field_name, field in pydantic_model.model_fields.items():
                self.logger.info(
                    f"  {field_name}: {'required' if field.is_required else 'optional'}" # type: ignore[truthy-function]
                )
                if field.default is not None:
                    self.logger.info(f"    Default: {field.default}")

            self.logger.info("\nModel Schema:")
            self.logger.info(pydantic_model.model_json_schema())
