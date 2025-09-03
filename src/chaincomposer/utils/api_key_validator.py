from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional
import warnings

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from ..types import ValidationResult

if TYPE_CHECKING:
    from langchain.schema.runnable import Runnable

class APIKeyValidator:
    """Validator for LLM API keys across different services.

    This class validates API keys for different LLM services like OpenAI, Anthropic,
    and Google. It caches validation results to avoid redundant validation checks.

    Attributes:
        _validated_already (Dict[str, ValidationResult]): A dictionary that stores
            validation results for each API key, where the key is the API key string
            and the value is a ValidationResult object containing validation status
            for each service.

    Examples:
        Basic usage of the validator:

            >>> validator = APIKeyValidator()
            >>> if validator.is_valid(api_key="sk-...", model="gpt-4"):
            ...     print("Key is valid!")

        The validator caches results for subsequent checks:

            >>> # First check validates against services
            >>> validator.is_valid("sk-...", "gpt-4")
            >>> # Second check uses cached result
            >>> validator.is_valid("sk-...", "gpt-4")
    """

    def __init__(self):
        """Initialize the APIKeyValidator class."""
        # Dict to track api keys which have been validated already
        # Key = api_key, Value = ValidationResult
        self._validated_already: Dict[str, ValidationResult] = {}
        
    def is_valid(self, api_key: str, model: Optional[str] = None) -> bool:
        """Check if the API key is valid for any service. Validates if not already done."""
        if api_key not in self._validated_already:
            self._validate(api_key=api_key, model=model)

        results = self._validated_already[api_key]
        return any([results.openai, results.anthropic, results.google])
    
    def get_results_by_key_as_object(self, api_key: str) -> ValidationResult:
        """Get detailed validation results. Validates if not already done.
        
        Returns:
            ValidationResult: The validation results.
        """
        if api_key not in self._validated_already:
            raise ValueError(
                f"API key {self._mask(api_key)} has not been validated yet. "
                "Please validate the API key before calling this method.\n"
                "You can do so by calling the `is_valid` method on an APIKeyValidator instance.\n"
                ">>> validator = APIKeyValidator()\n"
                ">>> is_valid: bool = validator.is_valid(api_key, model)\n\n"
                "Then you can call the `get_results_by_key_as_object` method to recieve a ValidationResult object. "
                "This object is a dataclass with the following attributes:\n"
                "openai: bool\n"
                "anthropic: bool\n"
                "google: bool\n\n"
                "Usage Example:\n"
                ">>> results: ValidationResult = validator.get_results_by_key_as_object(api_key)\n"
                ">>> print(results.openai)\n"
                ">>> print(results.anthropic)\n"
                ">>> print(results.google)\n\n"
            )

        return self._validated_already[api_key]

    def get_results_for_api_key(self, api_key: str) -> Dict[str, bool]:
        """Get detailed validation results. Validates if not already done."""
        if api_key not in self._validated_already:
            raise ValueError(
                f"API key {self._mask(api_key)} has not been validated yet. "
                "Please validate the API key before calling this method.\n"
                "You can do so by calling the `is_valid` method on an APIKeyValidator instance.\n"
                ">>> validator = APIKeyValidator()\n"
                ">>> is_valid: bool = validator.is_valid(api_key, model)\n\n"
                "Then you can call the `get_results_for_api_key` method to recieve a dictionary of results. "
                "This dictionary has the following keys:\n"
                "openai: bool\n"
                "anthropic: bool\n"
                "google: bool\n\n"
                "Usage Example:\n"
                ">>> results: Dict[str, bool] = validator.get_results_for_api_key(api_key)\n"
                ">>> print(results['openai'])\n"
                ">>> print(results['anthropic'])\n"
                ">>> print(results['google'])\n\n"
            )
        
        results = self._validated_already[api_key]
        
        return {
            "openai": results.openai,
            "anthropic": results.anthropic,
            "google": results.google,
        }

    def _validate(
        self, 
        api_key: str, 
        model: Optional[str] = None,
    ) -> None:
        """Run validation tests for each service."""
        
        results: ValidationResult = ValidationResult()
        prompt: ChatPromptTemplate = self._get_test_prompt()
        
        # Store warnings instead of immediately showing them
        warnings_list = []
        
        def test_with_warning_capture(provider_name: str, provider_func) -> bool:
            result, warning = self._test_provider(
                api_key,
                provider_func(api_key, model),
                prompt,
                model,
            )
            if not result and warning:
                warnings_list.append(f"{provider_name}: {warning}")
            return result
            
        # Test each provider
        results.openai = test_with_warning_capture("OpenAI", self._get_openai_llm)
        results.anthropic = test_with_warning_capture("Anthropic", self._get_anthropic_llm)
        results.google = test_with_warning_capture("Google", self._get_google_llm)
        
        # Only show warnings if all providers failed
        if not any([results.openai, results.anthropic, results.google]):
            for warning in warnings_list:
                warnings.warn(warning)
        
        self._validated_already[api_key] = results
        
    def _get_test_prompt(self) -> ChatPromptTemplate:
        """Get a test prompt for the API key validator.
        
        Returns:
            ChatPromptTemplate: The test prompt.
        """
        system_prompt_template: PromptTemplate = PromptTemplate(template="test")
        human_prompt_template: PromptTemplate = PromptTemplate(template="test")

        prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    system_prompt_template.template
                ),
                HumanMessagePromptTemplate.from_template(
                    human_prompt_template.template
                ),
            ]
        )
        
        return prompt
    
    def _get_openai_llm(
        self, 
        api_key: str, 
        model: Optional[str] = None,
    ) -> ChatOpenAI:
        """Get an OpenAI LLM instance."""
        if model is None:
            return ChatOpenAI(api_key=api_key)
        else:
            return ChatOpenAI(api_key=api_key, model=model)
    
    def _get_anthropic_llm(
        self, 
        api_key: str, 
        model: Optional[str] = None,
    ) -> ChatAnthropic:
        """Get an Anthropic LLM instance."""
        if model is None:
            return ChatAnthropic(api_key=api_key)
        else:
            return ChatAnthropic(api_key=api_key, model=model)
    
    def _get_google_llm(
        self, 
        api_key: str, 
        model: Optional[str] = None,
    ) -> ChatGoogleGenerativeAI:
        """Get a Google Generative AI LLM instance."""
        if model is None:
            return ChatGoogleGenerativeAI(api_key=api_key)
        else:
            return ChatGoogleGenerativeAI(api_key=api_key, model=model)
    
    def _test_provider(
        self,
        api_key: str,
        llm: ChatOpenAI | ChatAnthropic | ChatGoogleGenerativeAI,
        prompt: ChatPromptTemplate,
        model: str | None = None,
    ) -> tuple[bool, str | None]:
        """Run a test to see if the API key is valid for a given provider.
        
        Args:
            api_key (str): The API key to test.
            llm (ChatOpenAI | ChatAnthropic | ChatGoogleGenerativeAI): The LLM to test.
            prompt (ChatPromptTemplate): The prompt to test the LLM with.
            model (str | None): The model to test the LLM with.

        Returns:
            tuple[bool, str | None]: A tuple containing a boolean indicating if the API key is valid, and a string containing a warning message if the API key is not valid.
        """
        try:
            chain: Runnable = prompt | llm
            chain.invoke({})
            return True, None
        except Exception as e:
            error_type = str(type(e).__name__)
            if any(err in error_type for err in [
                # OpenAI
                "AuthenticationError",
                "PermissionError",
                
                # Anthropic
                "AuthenticationError",
                
                # Google
                "PermissionDenied",
                "InvalidArgument",
                
                # General
                "RequestException",
            ]):
                warning_msg = (
                    f"The API key: {self._mask(api_key)} has an issue associated with it.\n"
                    f"Error: {str(e)}\n"
                    f"Provider: {type(llm)}\n"
                    f"Model: {model}\n"
                )
                return False, warning_msg
            else:
                warning_msg = f"Unexpected error during API key validation: {str(e)}"
                return False, warning_msg
        
    def _mask(self, api_key: str) -> str:
        """Mask the API key for display purposes."""
        return api_key[:4] + "..." + api_key[-4:]
