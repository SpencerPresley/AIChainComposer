from typing import (
    Any,
    Dict,
    TypeVar,
    Union,
    TypeAlias,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from langchain_core.output_parsers import (
        JsonOutputParser,
        PydanticOutputParser,
        StrOutputParser,
    )
    
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic

FirstCallRequired = TypeVar("FirstCallRequired", bound=Dict[str, Any])
"""TypeVar representing a dictionary that is required on first call but optional after.

Type Structure:
    TypeVar bound to Dict[str, Any]

Usage:
    Used to enforce that a dictionary parameter must be provided on first call
    but can be optional on subsequent calls.
"""

ParserUnion: TypeAlias = Union[PydanticOutputParser, JsonOutputParser, StrOutputParser]
"""Union type representing valid parser types.

Type Structure:
    Union[PydanticOutputParser, JsonOutputParser, StrOutputParser]
"""

ParserType = TypeVar("ParserType", bound=ParserUnion)
"""TypeVar representing the main parser type.

Type Structure:
    TypeVar bound to Union[PydanticOutputParser, JsonOutputParser, StrOutputParser]

Usage:
    Used to enforce type consistency for the primary parser in chain operations.
"""

FallbackParserType = TypeVar("FallbackParserType", bound=ParserUnion)
"""TypeVar representing the fallback parser type.

Type Structure:
    TypeVar bound to Union[PydanticOutputParser, JsonOutputParser, StrOutputParser]

Usage:
    Used to enforce type consistency for the fallback parser in chain operations.
"""

LLMUnion = TypeVar("LLMUnion", bound=Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI])
"""TypeVar representing the language model type.

Type Structure:
    TypeVar bound to Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI]

Usage:
    Used to enforce type consistency for the language model in chain operations.
"""
