INPUT_ERROR_MESSAGE: str = (
    "No input data provided to ChainWrapper.run_chain().\n"
    "Input data is required to run the chain.\n"
)

MAIN_CHAIN_JSON_ERROR_MESSAGE: str = (
    "A json decode error occurred in the main chain. Consider:\n"
    "1. Check your JSON format instructions\n"
    "2. Use StrOutputParser() for intermediate chains\n"
    "3. Simplify your Pydantic model\n"
    "4. Try a more capable model\n"
)

MAIN_CHAIN_VALIDATION_ERROR_MESSAGE: str = (
    "A validation error occurred in the main chain. Consider:\n"
    "1. Check your Pydantic model\n"
    "2. Simplify your Pydantic model\n"
    "3. Try a more capable model\n"
)

MAIN_CHAIN_VALUE_ERROR_MESSAGE: str = (
    "A value error occurred in the main chain. Consider:\n"
    "1. Check your input data\n"
    "2. Simplify your Pydantic model\n"
    "3. Try a more capable model\n"
)

MAIN_CHAIN_TYPE_ERROR_MESSAGE: str = (
    "A type error occurred in the main chain. Consider:\n"
    "1. Check your input data\n"
    "2. Simplify your Pydantic model\n"
    "3. Try a more capable model\n"
)

FALLBACK_CHAIN_JSON_ERROR_MESSAGE: str = (
    "A json decode error occurred in the fallback chain. Consider:\n"
    "1. Check your JSON format instructions\n"
    "2. Use StrOutputParser() for intermediate chains\n"
    "3. Simplify your Pydantic model\n"
    "4. Try a more capable model\n"
)

FALLBACK_CHAIN_VALIDATION_ERROR_MESSAGE: str = (
    "A validation error occurred in the fallback chain. Consider:\n"
    "1. Check your Pydantic model\n"
    "2. Simplify your Pydantic model\n"
    "3. Try a more capable model\n"
)

FALLBACK_CHAIN_VALUE_ERROR_MESSAGE: str = (
    "A value error occurred in the fallback chain. Consider:\n"
    "1. Check your input data\n"
    "2. Simplify your Pydantic model\n"
    "3. Try a more capable model\n"
)

FALLBACK_CHAIN_TYPE_ERROR_MESSAGE: str = (
    "A type error occurred in the fallback chain. Consider:\n"
    "1. Check your input data\n"
    "2. Simplify your Pydantic model\n"
    "3. Try a more capable model\n"
)
