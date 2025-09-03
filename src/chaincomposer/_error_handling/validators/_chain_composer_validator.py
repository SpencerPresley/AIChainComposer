from __future__ import annotations

from typing import Union, TYPE_CHECKING, cast, Type

from ..enums._error_reference import _ChainComposerErrorReference
from ..factories import _ChainExceptionFactory

if TYPE_CHECKING:
    from pydantic import BaseModel
    from chain_composer._type_aliases import FirstCallRequired

class _ChainComposerValidator:
    """Validator class for ChainComposer operations."""
    
    @staticmethod
    def validate_llm_model_type(llm_model: str) -> str:
        """Validate and determine the LLM model type.
        
        Args:
            llm_model (str): The name of the LLM model.
            
        Returns:
            str: The validated model type.
            
        Raises:
            ChainError: If the model type is not supported.
        """
        if llm_model.lower().startswith("gpt"):
            return "openai"
        elif llm_model.lower().startswith("claude"):
            return "anthropic"
        elif llm_model.lower().startswith("gemini"):
            return "google"
        
        raise _ChainExceptionFactory.create_error(
            error=ValueError(f"Unsupported LLM model: {llm_model}"),
            error_reference=_ChainComposerErrorReference.UNSUPPORTED_LLM_MODEL
        )
    
    @staticmethod
    def validate_parser_configuration(
        *,
        output_passthrough_key_name: str | None,
        ignore_output_passthrough_key_name_error: bool,
        parser_type: str | None,
        pydantic_output_model: Type[BaseModel] | None,
        fallback_parser_type: str | None,
        fallback_pydantic_output_model: Type[BaseModel] | None,
    ) -> None:
        """Validate parser configuration parameters.
        
        Args:
            output_passthrough_key_name (str | None): Key name for output.
            ignore_output_passthrough_key_name_error (bool): Whether to ignore missing key name.
            parser_type (str | None): Type of parser.
            pydantic_output_model (Type[BaseModel] | None): Pydantic model for output.
            fallback_parser_type (str | None): Type of fallback parser.
            fallback_pydantic_output_model (Type[BaseModel] | None): Pydantic model for fallback.
            
        Raises:
            ChainError: If validation fails.
        """
        if not output_passthrough_key_name and not ignore_output_passthrough_key_name_error:
            raise _ChainExceptionFactory.create_error(
                error=ValueError("Missing output key name"),
                error_reference=_ChainComposerErrorReference.MISSING_OUTPUT_KEY
            )
            
        if parser_type is None and fallback_parser_type is not None:
            raise _ChainExceptionFactory.create_error(
                error=ValueError("Invalid parser combination"),
                error_reference=_ChainComposerErrorReference.INVALID_PARSER_COMBINATION
            )
            
        if parser_type == "pydantic" and not pydantic_output_model:
            raise _ChainExceptionFactory.create_error(
                error=ValueError("Missing Pydantic model for parser"),
                error_reference=_ChainComposerErrorReference.MISSING_PYDANTIC_MODEL
            )
            
        if fallback_parser_type == "pydantic" and not fallback_pydantic_output_model:
            raise _ChainExceptionFactory.create_error(
                error=ValueError("Missing Pydantic model for fallback parser"),
                error_reference=_ChainComposerErrorReference.MISSING_PYDANTIC_MODEL
            )
            
        if parser_type and parser_type not in ["pydantic", "json", "str"]:
            raise _ChainExceptionFactory.create_error(
                error=ValueError(f"Invalid parser type: {parser_type}"),
                error_reference=_ChainComposerErrorReference.INVALID_PARSER_TYPE
            )
            
        if fallback_parser_type and fallback_parser_type not in ["pydantic", "json", "str"]:
            raise _ChainExceptionFactory.create_error(
                error=ValueError(f"Invalid fallback parser type: {fallback_parser_type}"),
                error_reference=_ChainComposerErrorReference.INVALID_PARSER_TYPE
            )
    
    @staticmethod
    def validate_prompt_variables(
        prompt_variables_dict: Union[FirstCallRequired, None],
        is_first_call: bool,
    ) -> None:
        """Validate prompt variables.
        
        Args:
            prompt_variables_dict (Union[FirstCallRequired, None]): The variables dict.
            is_first_call (bool): Whether this is the first call to run().
            
        Raises:
            ChainError: If validation fails.
        """
        if is_first_call and prompt_variables_dict is None:
            raise _ChainExceptionFactory.create_error(
                error=ValueError("First call must provide prompt variables"),
                error_reference=_ChainComposerErrorReference.MISSING_PROMPT_VARIABLES
            )
            
        if prompt_variables_dict is not None:
            if not isinstance(cast(dict, prompt_variables_dict), dict):
                raise _ChainExceptionFactory.create_error(
                    error=TypeError(
                        "prompt_variables_dict must be a dictionary. "
                        "Each key should match the variable names used in your chain layers. "
                        "output_passthrough_key_name parameter in add_chain_layer method is used to identify the output "
                        "of the chain layer "
                        "and assign it to a variable. If you do not specify output_passthrough_key_name, the output of "
                        "the chain layer will not be assigned to a variable and thus will not be available to the next "
                        "chain layer. "
                        "If you do not need the output of the chain layer to be passed to the next chain layer, you can "
                        "set ignore_output_passthrough_key_name_error to True. "
                        "A time to set ignore_output_passthrough_key_name_error to True is when you are running a chain "
                        "layer solely for its side effects (e.g. printing, saving to a database, etc.) without needing the "
                        "output of the chain layer to be passed to the next chain layer. "
                        "Another reason to set ignore_output_passthrough_key_name_error to True is if you have a "
                        "multi-layer chain and this is your last chain layer. "
                        "Check your prompt strings for your placeholder variables, these names should match the keys in "
                        f"{prompt_variables_dict} passed into the ChainManager.run() method."
                    ),
                    error_reference=_ChainComposerErrorReference.INVALID_PROMPT_VARIABLES
                ) 
