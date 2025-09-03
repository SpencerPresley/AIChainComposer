from enum import Enum

class _BaseErrorReference(Enum):
    pass

class _ChainBuilderErrorReference(_BaseErrorReference):
    pass

class _ChainManagerErrorReference(_BaseErrorReference):
    pass

class _ChainWrapperErrorReference(_BaseErrorReference):
    INPUT_ERROR = "input_error"
    MAIN_CHAIN_ERROR = "main_chain_error"
    FALLBACK_CHAIN_ERROR = "fallback_chain_error"

class _ChainComposerErrorReference(_BaseErrorReference):
    UNSUPPORTED_LLM_MODEL = "unsupported_llm_model"
    INVALID_PARSER_TYPE = "invalid_parser_type"
    MISSING_PYDANTIC_MODEL = "missing_pydantic_model"
    MISSING_OUTPUT_KEY = "missing_output_key"
    INVALID_PARSER_COMBINATION = "invalid_parser_combination"
    MISSING_PROMPT_VARIABLES = "missing_prompt_variables"
    INVALID_PROMPT_VARIABLES = "invalid_prompt_variables"
    INVALID_API_KEY = "invalid_api_key"
