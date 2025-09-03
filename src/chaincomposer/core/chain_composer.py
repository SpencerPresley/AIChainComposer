from __future__ import annotations

import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
    Type,
)

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.runnable import Runnable
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from .._logging import get_logger, WARNING
from .._error_handling import ( # type: ignore[attr-defined]
    _ChainExceptionFactory,
    _ChainComposerErrorReference,
    _ChainComposerValidator,
)

from .._core_internal import (
    _ChainManager,
    _ChainWrapper,
    _ChainBuilder,
)

from ..utils import APIKeyValidator

if TYPE_CHECKING:
    from pydantic import BaseModel
    from .._type_aliases import (
        FirstCallRequired,
        ParserUnion,
    )


class ChainComposer:
    """Class responsible for managing and orchestrating a sequence of chain layers,

    ChainComposer is a class responsible for managing and orchestrating a sequence of chain layers,
    each of which can process input data and produce output data. It supports various types of
    language models (LLMs) and parsers, and allows for pre-processing and post-processing of data.

    Attributes:
        api_key (str): The API key for accessing the LLM service.
        llm_model (str): The name of the LLM model to use.
        llm_model_type (str): The type of the LLM model (e.g., "openai", "anthropic", "google").
            Type: Literal["openai", "anthropic", "google"]
        llm_temperature (float): The temperature setting for the LLM model.
        llm_kwargs (dict): Additional keyword arguments for the LLM model.
            Type: Dict[str, Any]
        llm (LLMUnion): The initialized LLM instance.
        chain_manager (_ChainManager): The manager for managing the sequence of chain layers.
        chain_variables (dict): A dictionary of variables used in the chain layers.
            Type: Dict[str, Any]
        chain_variables_update_overwrite_warning_counter (int): Counter for tracking variable overwrite warnings.
        preprocessor (Callable[[Dict[str, Any]], Dict[str, Any]] | None): Optional preprocessor function.
        postprocessor (Callable[[Any], Any] | None): Optional postprocessor function.
        logger (logging.Logger): Logger for logging information and debugging.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        preprocessor: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
        postprocessor: Callable[[Any], Any] | None = None,
        enable_logging: bool | None = False,
        level: int | None = WARNING,
        debug: bool = False,
    ) -> None:
        """Initializes the ChainComposer class.

        Args:
            llm_model (str): The name of the language model to be used.
            api_key (str): The API key for accessing the language model.
            llm_temperature (float, optional): The temperature setting for the language model.
                Defaults to 0.7.
            preprocessor (Callable[[Dict[str, Any]], Dict[str, Any]] | None): A function to preprocess input data.
                Defaults to None.
            postprocessor (Callable[[Any], Any] | None): A function to postprocess output data.
                Defaults to None.
            llm_kwargs (dict | None): Additional keyword arguments for the language model.
                Type: Dict[str, Any]
            words_to_ban (list): A list of words to ban from the language model's output.
                Type: List[str]
                Defaults to None.
            enable_logging (bool | None): Flag to enable logging.
                Defaults to False.
            level (logging.Level | None): The logging level.
                Defaults to WARNING.
        """
        self.logger = get_logger(
            module_name=__name__,
            level=level,
            null_logger=not enable_logging,
        )

        self.api_key: str = api_key
        self.llm_model: str = model
        self.llm_model_type: str = self._get_llm_model_type(llm_model=model)

        self.llm_temperature: float = temperature
        
        # Validate API key
        self._validate_api_key(api_key=api_key)
        
        self.llm: Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI] = (
            self._initialize_llm(
                api_key=self.api_key,
                llm_model_type=self.llm_model_type,
                llm_model=self.llm_model,
                llm_temperature=self.llm_temperature,
            )
        )
        self.enable_logging: bool = ( # type: ignore[assignment]
            False
            if (
                (enable_logging is False) or 
                (enable_logging is None) or 
                (not isinstance(enable_logging, bool))
            )
            else enable_logging
        )
        self.level: int = (  # type: ignore[assignment]
            WARNING
            if (
                (level is None) or 
                (not isinstance(level, int))
            )
            else level
        )
        self.chain_manager: _ChainManager = _ChainManager(
            enable_logging=self.enable_logging,
            level=self.level,
        )

        self.chain_variables: Dict[str, Any] = {}

        self.chain_variables_update_overwrite_warning_counter: int = 0
        self.preprocessor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = (
            preprocessor
        )
        self.postprocessor: Optional[Callable[[Any], Any]] = postprocessor
        self.debug: bool = debug

    def __str__(self) -> str:
        """Returns a concise string representation of the ChainComposer object.

        Returns:
            str: A concise string representation of the ChainComposer object.
        """
        return (
            f"ChainComposer(\n"
            f"  llm={self.llm.__class__.__name__},\n"
            f"  model={self.llm_model},\n"
            f"  temperature={self.llm_temperature},\n"
            f"  chain_count={len(self.chain_manager.chain_sequence)}\n"
            f")"
        )

    def __repr__(self) -> str:
        """Returns a detailed string representation for debugging.

        Returns:
            str: A detailed string representation of the ChainComposer object.
        """
        return (
            f"ChainComposer(\n"
            f"    llm={self.llm!r},\n"
            f"    llm_model={self.llm_model!r},\n"
            f"    llm_model_type={self.llm_model_type!r},\n"
            f"    llm_temperature={self.llm_temperature!r},\n"
            f"    chain_manager={self.chain_manager!r},\n"
            f"    chain_variables={self.chain_variables!r},\n"
            f"    preprocessor={self.preprocessor!r},\n"
            f"    postprocessor={self.postprocessor!r}\n"
            f")"
        )
    
    def add_chain_layer(
        self,
        *,
        system_prompt: str,
        human_prompt: str,
        output_passthrough_key_name: str | None = None,
        ignore_output_passthrough_key_name_error: bool = False,
        preprocessor: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
        postprocessor: Callable[[Any], Any] | None = None,
        parser_type: Literal["pydantic", "json", "str"] | None = None,
        fallback_parser_type: Literal["pydantic", "json", "str"] | None = None,
        pydantic_output_model: Type[BaseModel] | None = None,
        fallback_pydantic_output_model: Type[BaseModel] | None = None,
    ) -> "ChainComposer":
        """Adds a chain layer to the chain composer.

        This method configures and adds a new chain layer to the chain composer,
        allowing for the processing of input data through specified prompts and parsers.

        Args:
            system_prompt (str): The system prompt template for the chain layer.
            human_prompt (str): The human prompt template for the chain layer.
            output_passthrough_key_name (str | None): Key name for passing chain output
                to the next layer.
                Defaults to None.
            ignore_output_passthrough_key_name_error (bool): Flag to ignore missing output
                key name errors.
                Defaults to False.
            preprocessor (Callable[[Dict[str, Any]], Dict[str, Any]] | None): Function to preprocess input data.
                Defaults to None.
            postprocessor (Callable[[Any], Any] | None): Function to postprocess output data.
                Defaults to None.
            parser_type (str | None): Type of parser to use.
                Type: Literal["pydantic", "json", "str"] | None
                Defaults to None.
            fallback_parser_type (str | None): Type of fallback parser.
                Type: Literal["pydantic", "json", "str"] | None
                Defaults to None.
            pydantic_output_model (Type[BaseModel] | None): Pydantic model for output validation.
                Defaults to None
            fallback_pydantic_output_model (Type[BaseModel] | None): Pydantic model for fallback parser.
                Defaults to None.

        Returns:
            None
        """
        self._run_chain_validation_checks(
            output_passthrough_key_name=output_passthrough_key_name,
            ignore_output_passthrough_key_name_error=ignore_output_passthrough_key_name_error,
            parser_type=parser_type,
            pydantic_output_model=pydantic_output_model,
            fallback_parser_type=fallback_parser_type,
            fallback_pydantic_output_model=fallback_pydantic_output_model,
        )

        parser: Optional[ParserUnion] = None
        fallback_parser: Optional[ParserUnion] = None
        if parser_type:
            parser = self._initialize_parser(
                parser_type=parser_type, pydantic_output_model=pydantic_output_model
            )
        if fallback_parser_type:
            fallback_parser = self._initialize_parser(
                parser_type=fallback_parser_type,
                pydantic_output_model=fallback_pydantic_output_model,
            )
        # Create prompt templates without specifying input_variables
        system_prompt_template: PromptTemplate = PromptTemplate(template=system_prompt) # type: ignore[call-arg]
        human_prompt_template: PromptTemplate = PromptTemplate(template=human_prompt) # type: ignore[call-arg]
        system_message_prompt_template: SystemMessagePromptTemplate = (
            SystemMessagePromptTemplate.from_template(system_prompt_template.template)
        )
        human_message_prompt_template: HumanMessagePromptTemplate = (
            HumanMessagePromptTemplate.from_template(human_prompt_template.template)
        )

        chat_prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [system_message_prompt_template, human_message_prompt_template]
        )
        # Build the chain using ChainBuilder
        chain_builder: _ChainBuilder = _ChainBuilder(
            chat_prompt=chat_prompt_template,
            llm=self.llm,
            parser=parser,
            fallback_parser=fallback_parser,
            enable_logging=self.enable_logging,
            level=self.level,
        )
        chain: Runnable = chain_builder.get_chain()
        fallback_chain: Runnable = chain_builder.get_fallback_chain()

        # Wrap the chain
        chain_wrapper: _ChainWrapper = _ChainWrapper(
            chain=chain,
            fallback_chain=fallback_chain,
            parser=parser,
            fallback_parser=fallback_parser,
            preprocessor=preprocessor or self.preprocessor,
            postprocessor=postprocessor or self.postprocessor,
            enable_logging=self.enable_logging,
            level=self.level,
        )

        # Add the chain to the composer
        self.chain_manager.add_chain(
            chain_wrapper=chain_wrapper,
            output_passthrough_key_name=output_passthrough_key_name,
        )

        # Return the ChainManager instance to allow for method chaining
        return self
    
    def run(
        self,
        prompt_variables_dict: Union[FirstCallRequired, None] = None,
    ) -> Dict[str, Any]:
        """Executes the chain builder process.

        This method performs validation checks, updates chain variables if provided,
        and runs the chain composer with the current chain variables.

        Args:
            prompt_variables_dict (dict | None): A dictionary containing prompt variables.
                If provided, it will be used to update the chain variables.
                Type: Union[FirstCallRequired, None]
                Defaults to None.

        Returns:
            dict: The result of running the chain composer.
        """
        self._run_validation_checks(
            prompt_variables_dict=prompt_variables_dict,
        )

        if prompt_variables_dict is not None:
            self._update_chain_variables(prompt_variables_dict)

        return self.chain_manager.run(
            data_dict=self.chain_variables,
            data_dict_update_function=self._update_chain_variables,
        )
        
    def get_chain_sequence(self) -> List[Tuple[_ChainWrapper, str | None]]:
        """Retrieves the chain sequence from the chain composer.

        Returns:
            list: A list of tuples where each tuple contains a ChainWrapper object and
                the output key name if output_passthrough_key_name was provided to add_chain_layer.
                Type: List[Tuple[ChainWrapper, str | None]]
        """
        return self.chain_manager.chain_sequence

    def print_chain_sequence(self) -> None:
        """Prints the chain sequence by formatting it.

        This method retrieves the chain sequence from the chain composer and
        formats it using the _format_chain_sequence method.

        Returns:
            None
        """
        chain_sequence: List[Tuple[_ChainWrapper, str | None]] = (
            self.chain_manager.chain_sequence
        )
        self._format_chain_sequence(chain_sequence)

    def get_chain_variables(self) -> Dict[str, Any]:
        """Retrieve the chain variables.

        Returns:
            dict: A dictionary containing the chain variables.
                Type: Dict[str, Any]
        """
        return self.chain_variables

    def print_chain_variables(self) -> None:
        """Prints the chain variables in a formatted manner.

        This method prints the chain variables stored in the `chain_variables`
        attribute of the class. The output is formatted with a header and
        footer consisting of dashes, and each key-value pair is printed on
        a new line.

        Returns:
            None
        """
        print(f"Chain Variables:\n{'-' * 10}")
        for key, value in self.chain_variables.items():
            print(f"{key}: {value}")
        print(f"{'-' * 10}\n")

    def _get_llm_model_type(self, *, llm_model: str) -> str:
        """Determine the type of LLM (Large Language Model) based on the provided model name.

        Args:
            llm_model (str): The name of the LLM model.

        Returns:
            str: The type of the LLM model. Possible values are "openai", "anthropic", and "google".

        Raises:
            ChainError: If the provided LLM model name does not match any of the supported types.
        """
        return _ChainComposerValidator.validate_llm_model_type(llm_model)

    def _initialize_llm(
        self,
        *,
        api_key: str | None = None,
        llm_model_type: str | None = None,
        llm_model: str | None = None,
        llm_temperature: float | None = None,
    ) -> Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI]:
        """Initializes a language model based on the specified type.

        Args:
            api_key (str | None): The API key for accessing the language model service.
                Defaults to None.
            llm_model_type (str | None): The type of the language model (e.g., "openai", "anthropic", "google").
                Defaults to None.
            llm_model (str | None): The specific model to use within the chosen type.
                Defaults to None.
            llm_temperature (float | None): The temperature setting for the language model, affecting randomness.
                Defaults to None.

        Returns:
            LLMUnion: An instance of the initialized language model.

        Raises:
            ChainError: If the specified `llm_model_type` is not supported.
        """
        if api_key is None:
            api_key = self.api_key

        if llm_model_type is None:
            llm_model_type = self.llm_model_type

        if llm_model is None:
            llm_model = self.llm_model

        if llm_temperature is None:
            llm_temperature = self.llm_temperature

        if llm_model_type == "openai":
            return self._create_openai_llm(
                api_key, llm_model, llm_temperature
            )
        elif llm_model_type == "anthropic":
            return self._create_anthropic_llm(
                api_key, llm_model, llm_temperature
            )
        elif llm_model_type == "google":
            return self._create_google_llm(
                api_key, llm_model, llm_temperature
            )
        else:
            raise _ChainExceptionFactory.create_error(
                error=ValueError(
                    f"Unsupported LLM model type: {llm_model_type}. Supported types: openai, anthropic, google."
                ),
                error_reference=_ChainComposerErrorReference.UNSUPPORTED_LLM_MODEL_TYPE
            )

    def _recreate_llm(self) -> None:
        """Recreates the LLM with the current parameters."""
        self.llm = self._initialize_llm(
            api_key=self.api_key,
            llm_model_type=self.llm_model_type,
            llm_model=self.llm_model,
            llm_temperature=self.llm_temperature,
        )

    def _create_openai_llm(
        self,
        api_key: str,
        llm_model: str,
        llm_temperature: float,
    ) -> ChatOpenAI:
        """Creates an instance of the ChatOpenAI language model.

        Args:
            api_key (str): The API key for authenticating with the OpenAI service.
            llm_model (str): The identifier of the language model to use.
            llm_temperature (float): The temperature setting for the language model,
                which controls the randomness of the output.

        Returns:
            ChatOpenAI: An instance of the ChatOpenAI language model configured with
                the specified parameters.
        """
        return ChatOpenAI(
            model=llm_model,
            api_key=api_key, # type: ignore[arg-type]
            temperature=llm_temperature,
        )

    def _create_anthropic_llm(
        self, 
        api_key: str, 
        llm_model: str, 
        llm_temperature: float,
    ) -> ChatAnthropic:
        """Creates an instance of the ChatAnthropic language model.

        Args:
            api_key (str): The API key for authenticating with the Anthropic service.
            llm_model (str): The identifier of the language model to use.
            llm_temperature (float): The temperature setting for the language model,
                controlling the randomness of the output.

        Returns:
            ChatAnthropic: An instance of the ChatAnthropic language model.
        """
        return ChatAnthropic( # type: ignore[call-arg]
            model_name=llm_model, 
            api_key=api_key, # type: ignore[arg-type]
            temperature=llm_temperature,
        )

    def _create_google_llm(
        self, 
        api_key: str, 
        llm_model: str, 
        llm_temperature: float,
    ) -> ChatGoogleGenerativeAI:
        """Creates an instance of ChatGoogleGenerativeAI with the specified parameters.

        Args:
            api_key (str): The API key for authenticating with the Google LLM service.
            llm_model (str): The model identifier for the Google LLM.
            llm_temperature (float): The temperature setting for the LLM,
                which controls the randomness of the output.

        Returns:
            ChatGoogleGenerativeAI: An instance of the ChatGoogleGenerativeAI class configured
                with the provided parameters.
        """
        return ChatGoogleGenerativeAI(
            model=llm_model, 
            api_key=api_key, # type: ignore[arg-type]
            temperature=llm_temperature,
        )

    def _initialize_parser(
        self,
        parser_type: Literal["pydantic", "json", "str"],
        pydantic_output_model: Type[BaseModel] | None = None,
    ) -> ParserUnion:
        """Initializes and returns a parser based on the specified parser type.

        Args:
            parser_type (str): The type of parser to initialize.
                Must be one of "pydantic", "json", or "str".
                Type: Literal["pydantic", "json", "str"]
            pydantic_output_model (BaseModel | None): The Pydantic model to use for the parser. Required if parser_type is "pydantic".
                Defaults to None.

        Returns:
            ParserUnion: An instance of the specified parser type.

        Raises:
            ChainError: If an invalid parser_type is provided.
        """
        if parser_type == "pydantic":
            parser: PydanticOutputParser = self._create_pydantic_parser( # type: ignore[no-redef]
                pydantic_output_model=pydantic_output_model
            )
            self.logger.debug(f"Created Pydantic parser: {parser}")
            return parser
        elif parser_type == "json":
            parser: JsonOutputParser = self._create_json_parser( # type: ignore[no-redef]
                pydantic_output_model=pydantic_output_model
            )
            self.logger.debug(f"Created JSON parser: {parser}")
            return parser
        elif parser_type == "str":
            parser: StrOutputParser = self._create_str_parser() # type: ignore[no-redef]
            self.logger.debug(f"Created Str parser: {parser}")
            return parser
        else:
            raise _ChainExceptionFactory.create_error(
                error=ValueError(f"Invalid parser_type: {parser_type}"),
                error_reference=_ChainComposerErrorReference.INVALID_PARSER_TYPE
            )

    def _create_pydantic_parser(
        self,
        pydantic_output_model: Type[BaseModel] | None,
    ) -> PydanticOutputParser:
        """Creates a Pydantic output parser.

        Args:
            pydantic_output_model (BaseModel | None): The Pydantic model to be used for parsing output.

        Returns:
            PydanticOutputParser: An instance of PydanticOutputParser initialized with
                the provided Pydantic model.

        Raises:
            ChainError: If pydantic_output_model is not provided.

        Notes:
            - `pydantic_output_model` must be provided for `parser_type` 'pydantic'.
        """
        if not pydantic_output_model:
            raise _ChainExceptionFactory.create_error(
                error=ValueError(
                    "pydantic_output_model must be provided for 'pydantic' parser_type."
                ),
                error_reference=_ChainComposerErrorReference.MISSING_PYDANTIC_MODEL
            )
        return PydanticOutputParser(pydantic_object=pydantic_output_model)

    def _create_json_parser(
        self, *, pydantic_output_model: Type[BaseModel] | None
    ) -> JsonOutputParser:
        """Creates a JSON parser for the chain layer output.

        Args:
            pydantic_output_model (Type[BaseModel] | None): An optional Pydantic model to enforce
                typing on the JSON output.
                Type: Type[BaseModel] | None
                Defaults to None.

        Returns:
            JsonOutputParser: An instance of JsonOutputParser. If pydantic_output_model is provided, the parser will enforce the model's schema on the output.

        Raises:
            UserWarning: If `pydantic_output_model` is not provided, a warning is issued
                recommending its use for proper typing of the output.
        """
        json_parser: JsonOutputParser | None = None
        if not pydantic_output_model:
            warnings.warn(
                "It is highly recommended to provide a pydantic_output_model when parser_type is 'json'. "
                "This will ensure that the output of the chain layer is properly typed and can be used in downstream chain layers."
            )
            self.logger.debug("Creating JSON parser without pydantic_output_model. ")
            json_parser = JsonOutputParser()
        else:
            self.logger.debug(
                f"Creating JSON parser with pydantic_output_model: {pydantic_output_model}"
            )
        return JsonOutputParser(pydantic_object=pydantic_output_model)

    def _create_str_parser(self) -> StrOutputParser:
        """Creates an instance of StrOutputParser.

        Returns:
            StrOutputParser: An instance of the StrOutputParser class.
        """
        return StrOutputParser()

    def _run_chain_validation_checks(
        self,
        *,
        output_passthrough_key_name: str | None,
        ignore_output_passthrough_key_name_error: bool,
        parser_type: Literal["pydantic", "json", "str"] | None,
        pydantic_output_model: Type[BaseModel] | None,
        fallback_parser_type: Literal["pydantic", "json", "str"] | None,
        fallback_pydantic_output_model: Type[BaseModel] | None,
    ) -> None:
        """Validates chain configuration parameters before execution.

        Performs validation checks on chain configuration parameters to ensure proper setup
        and compatibility between different components.

        Args:
            output_passthrough_key_name (str | None): Optional key name for passing chain
                output to next layer.
                Defaults to None.
            ignore_output_passthrough_key_name_error (bool): Whether to ignore missing output
                key name errors.
                Defaults to False.
            parser_type (str | None): Type of parser to use.
                Type: Literal["pydantic", "json", "str"] | None
                Defaults to None.
            pydantic_output_model (Type[BaseModel] | None): Pydantic model for output validation.
                Defaults to None.
            fallback_parser_type (str | None): Type of fallback parser.
                Type: Literal["pydantic", "json", "str"] | None
                Defaults to None.
            fallback_pydantic_output_model (Type[BaseModel] | None): Pydantic model for fallback parser.
                Defaults to None.

        Raises:
            ChainError: If validation fails for:
                - Missing output key name when required
                - Invalid parser type combinations
                - Missing required models
                - Duplicate parser types
                - Same models used for main and fallback

        Warnings:
            UserWarning: For non-critical issues like:
                - Missing output key name when ignored
                - Missing recommended Pydantic models
                - Unused provided models
        """
        _ChainComposerValidator.validate_parser_configuration(
            output_passthrough_key_name=output_passthrough_key_name,
            ignore_output_passthrough_key_name_error=ignore_output_passthrough_key_name_error,
            parser_type=parser_type,
            pydantic_output_model=pydantic_output_model,
            fallback_parser_type=fallback_parser_type,
            fallback_pydantic_output_model=fallback_pydantic_output_model,
        )

    def _format_chain_sequence(
        self, chain_sequence: List[Tuple[_ChainWrapper, str | None]]
    ) -> None:
        """Formats and prints the details of each chain in the given chain sequence.

        Args:
            chain_sequence (list): A list of tuples where each tuple contains a ChainWrapper
                object and an optional output name.
                Type: List[Tuple[ChainWrapper, str | None]]

        Returns:
            None
        """
        for index, (chain_wrapper, output_name) in enumerate(chain_sequence):
            print(f"Chain {index + 1}:")
            print(f"\tOutput Name: {output_name}")
            print(f"\tParser Type: {chain_wrapper.get_parser_type()}")
            print(f"\tFallback Parser Type: {chain_wrapper.get_fallback_parser_type()}")
            print(f"\tPreprocessor: {chain_wrapper.preprocessor}")
            print(f"\tPostprocessor: {chain_wrapper.postprocessor}")

    def _run_validation_checks(
        self,
        *,
        prompt_variables_dict: Union[FirstCallRequired, None],
    ) -> None:
        """Validates the input parameters for the chain execution.

        Args:
            prompt_variables_dict (dict | None): A dictionary containing the variables
                to be passed to the chain layers.
                Type: Union[FirstCallRequired, None]
                - On the first call to `run()`, this parameter must be provided.
                - On subsequent calls, it can be omitted if there are no new variables to pass.

        Raises:
            ChainError: If `prompt_variables_dict` is None on the first call to `run()`.
            TypeError: If `prompt_variables_dict` is not a dictionary when provided.

        Notes:
            - The `prompt_variables_dict` should contain keys that match the variable names
              used in the chain layers.
            - The `output_passthrough_key_name` parameter in the `add_chain_layer` method is
              used to identify the output of the chain layer and assign it to a variable.
            - If `output_passthrough_key_name` is not specified, the output of the chain layer
              will not be assigned to a variable and will not be available to the next chain layer.
            - The `ignore_output_passthrough_key_name_error` parameter can be set to True if
              the output of the chain layer is not needed for the next chain layer, such as
              when running a chain layer solely for its side effects or if it is the last
              chain layer in a multi-layer chain.
            - Ensure that the placeholder variable names in your prompt strings match the keys
              in `prompt_variables_dict` passed into the `ChainManager.run()` method.
        """
        _ChainComposerValidator.validate_prompt_variables(
            prompt_variables_dict=prompt_variables_dict,
            is_first_call=self.chain_variables_update_overwrite_warning_counter == 0,
        )

    def _format_overwrite_warning(self, overwrites: Dict[str, Dict[str, Any]]) -> str:
        """Formats a warning message for overwritten values.

        Args:
            overwrites (dict): A dictionary where the key is the name of the overwritten item,
                and the value is another dictionary with 'old' and 'new' keys representing
                the old and new values respectively.
                Type: Dict[str, Dict[str, Any]]

        Returns:
            str: A formatted string that lists each overwritten item with its old and
                new values.
        """
        return "\n".join(
            f"  {key}:\n    - {details['old']}\n    + {details['new']}"
            for key, details in overwrites.items()
        )

    def _check_first_time_overwrites(
        self, prompt_variables_dict: Dict[str, Any]
    ) -> None:
        """Checks and warns if any global chain variables are being overwritten for the first time.

        This method compares the keys in the provided `prompt_variables_dict` with the existing
        `chain_variables`. If any keys match, it indicates that an overwrite is occurring. A warning
        is issued the first time this happens, detailing the old and new values of the overwritten
        variables. Subsequent overwrites will not trigger warnings.

        Args:
            prompt_variables_dict (dict): A dictionary containing the new values for the chain
                variables that may overwrite existing ones.
                Type: Dict[str, Any]

        Returns:
            None
        """
        if self.chain_variables_update_overwrite_warning_counter == 0:
            # overwrites = {
            #     key: {
            #         "old": self.chain_variables[key],
            #         "new": prompt_variables_dict[key],
            #     }
            #     for key in prompt_variables_dict
            #     if key in self.chain_variables
            # }
            # if overwrites:
            #     warnings.warn(
            #         f"Overwriting existing global variables:\n"
            #         f"{self._format_overwrite_warning(overwrites)}\n"
            #         "Subsequent overwrites will not trigger warnings."
            #     )
            self.chain_variables_update_overwrite_warning_counter += 1

    def _update_chain_variables(self, variables: Dict[str, Any]) -> None:
        """Update chain variables.

        Args:
            variables (Dict[str, Any]): The variables to update.
        """
        for key, value in variables.items():
            if key in self.chain_variables:
                if self.debug:
                    warnings.warn(
                        f"Overwriting existing chain variable '{key}' with new value '{value}'",
                        UserWarning,
                        stacklevel=2
                    )
            self.chain_variables[key] = value

    def _validate_api_key(self, api_key: str) -> None:
        """Validate the API key.

        Args:
            api_key (str): The API key to validate.

        Raises:
            ChainError: If the API key is invalid.
        """
        if not api_key:
            raise _ChainExceptionFactory.create_error(
                error=ValueError("API key cannot be empty"),
                error_reference=_ChainComposerErrorReference.INVALID_API_KEY
            )

        validator = APIKeyValidator()
        if not validator.is_valid(api_key=api_key, model=self.llm_model):
            raise _ChainExceptionFactory.create_error(
                error=ValueError("Invalid API key"),
                error_reference=_ChainComposerErrorReference.INVALID_API_KEY
            )

