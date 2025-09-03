from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Class to store the validation results. Used by APIKeyValidator.
    
    Attributes:
        openai (bool): True if the OpenAI API key is valid, False otherwise.
        anthropic (bool): True if the Anthropic API key is valid, False otherwise.
        google (bool): True if the Google API key is valid, False otherwise.
    """
    openai: bool = False
    anthropic: bool = False
    google: bool = False