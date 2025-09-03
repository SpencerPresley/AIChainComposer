import os
from dotenv import load_dotenv
load_dotenv()
from chaincomposer import ChainComposer
from pydantic import BaseModel

class FirstDerivative(BaseModel):
    first_derivative: str
    
class SecondDerivative(BaseModel):
    second_derivative: str

first_derivative_system_message = """
You are a helpful assistant, who takes the first derivative of a function and returns the result in the following format:

{{
"first_derivative": "<answer>"
}}

IMPORTANT: You should always return json. Do not include the markdown json format, just return the json.
"""

second_derivative_system_message = """
You are a helpful assistant, who takes a second derivative and returns the result in the following format:

{{
"second_derivative": "<answer>"
}}

IMPORTANT: You should always return json. Do not include the markdown json format, just return the json.
"""

human_message = """
Equation:

{equation}
"""

human_message_2 = """
Original Equation:

{equation}

First Derivative:

{first_derivative}
"""

def run_simple_example_json(model: str, api_key: str):
    cp = ChainComposer(
        model=model,
        api_key=api_key,
    )
    
    cp.add_chain_layer(
        system_prompt=first_derivative_system_message,
        human_prompt=human_message,
        output_passthrough_key_name="first_derivative",
        parser_type="json",
        pydantic_output_model=FirstDerivative
    ).add_chain_layer(
        system_prompt=second_derivative_system_message,
        human_prompt=human_message_2,
        output_passthrough_key_name="second_derivative",
        parser_type="json",
        pydantic_output_model=SecondDerivative
    )

    result = cp.run(
        prompt_variables_dict={
            "equation": "2x^2 + 3x + 2"
        }
    )

    print(result)

def run_simple_example_pydantic(model: str, api_key: str):
    cp = ChainComposer(
        model=model,
        api_key=api_key,
    )
    
    cp.add_chain_layer(
        system_prompt=first_derivative_system_message,
        human_prompt=human_message,
        output_passthrough_key_name="first_derivative",
        parser_type="pydantic",
        pydantic_output_model=FirstDerivative
    ).add_chain_layer(
        system_prompt=second_derivative_system_message,
        human_prompt=human_message_2,
        output_passthrough_key_name="second_derivative",
        parser_type="pydantic",
        pydantic_output_model=SecondDerivative
    )

    result = cp.run(
        prompt_variables_dict={
            "equation": "2x^2 + 3x + 2"
        }
    )

    print(result)

if __name__ == "__main__":
    openai_model = "gpt-5-nano"
    mistral_model = "mistral-large-latest"
    anthropic_model = "claude-3-5-haiku-20241022"
    gemini_model = "gemini-2.5-flash-lite"

    openai_api_key = os.getenv("OPENAI_API_KEY")
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    print(f"Running simple example with OpenAI model {openai_model}")
    print(f"=== OPENAI JSON ===\n\n")
    run_simple_example_json(openai_model, openai_api_key)
    print(f"\n\n=== OPENAI PYDANTIC ===\n\n")
    run_simple_example_pydantic(openai_model, openai_api_key)


    # TODO: Add this test once mistral is supported
    # print(f"\n\nRunning simple example with Mistral model {mistral_model}")
    # print(f"=== MISTRAL JSON ===\n\n")
    # run_simple_example_json(mistral_model, mistral_api_key)
    # print(f"\n\n=== MISTRAL PYDANTIC ===\n\n")
    # run_simple_example_pydantic(mistral_model, mistral_api_key)

    print(f"\n\nRunning simple example with Anthropic model {anthropic_model}")
    print(f"=== ANTHROPIC JSON ===\n\n")
    run_simple_example_json(anthropic_model, anthropic_api_key)
    print(f"\n\n=== ANTHROPIC PYDANTIC ===\n\n")
    run_simple_example_pydantic(anthropic_model, anthropic_api_key)

    print(f"\n\nRunning simple example with Gemini model {gemini_model}")
    print(f"=== GEMINI JSON ===\n\n")
    run_simple_example_json(gemini_model, gemini_api_key)
    print(f"\n\n=== GEMINI PYDANTIC ===\n\n")
    run_simple_example_pydantic(gemini_model, gemini_api_key)
    print(f"\n\nDONE")
