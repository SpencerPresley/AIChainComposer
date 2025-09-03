from dotenv import load_dotenv
import os
import json
from pydantic import BaseModel
from chaincomposer import ChainComposer

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

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

def with_json_parser():
    cp = ChainComposer(
        model="gpt-5-nano",
        api_key=api_key,
    )
    
    return cp.add_chain_layer(
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

def with_pydantic_parser():
    cp = ChainComposer(
        model="gpt-5-nano",
        api_key=api_key,
    )
    
    return cp.add_chain_layer(
        system_prompt=first_derivative_system_message,
        human_prompt=human_message,
        output_passthrough_key_name="first_derivative",
        parser_type="pydantic",
        pydantic_output_model=FirstDerivative
    ).add_chain_layer(
        system_prompt=second_derivative_system_message,
        human_prompt=human_message,
        output_passthrough_key_name="second_derivative",
        parser_type="pydantic",
        pydantic_output_model=SecondDerivative
    )

json_result = with_json_parser().run(
    prompt_variables_dict={
        "equation": "2x^2 + 3x + 2"
    }
)

pydantic_result = with_pydantic_parser().run(
    prompt_variables_dict={
        "equation": "2x^2 + 3x + 2"
    }
)

def print_json_result(result: dict):
    print(f"With json parser type on the final layer prior to running, the output is a dictionary:\n\nType: {type(result)}\n{json.dumps(result, indent=4)}\n")
    
def print_in_depth_pydantic_result(result: dict):
    print(f"\n\nFor the pydantic parser type for the final layer, the output dictionary will contain the pydantic model passed in as the value of the key passed in as the output_passthrough_key_name in the last layer.\n")
    print(f"Original Input:\n{result['equation']}\n")
    print(f"Output from first layer is sent to next as json, thus is json in output:\n\nType: {type(result['first_derivative'])}\n\n{result['first_derivative']}\n")
    print(f"Output from second layer is sent to next as json, but if the parser_type is \"pydantic\", it will be converted back to a pydantic model before being added back to the return dict:\n\nType: {type(result['second_derivative'])}\n\n{result['second_derivative']}")

print_json_result(json_result)
print_in_depth_pydantic_result(pydantic_result)



