from typing import Optional

import boto3
import llm
import re
import json
from pydantic import Field, field_validator

DEFAULT_LLAMA2_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()

DEFAULT_LLAMA3_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()


@llm.hookimpl
def register_models(register):
    register(
        BedrockLlama("meta.llama2-13b-chat-v1"),
        aliases=("bedrock-llama2-13b", 'bl2'),
    )
    register(
        BedrockLlama("meta.llama2-70b-chat-v1"),
        aliases=("bedrock-llama2-70b-chat-v1", "bl2-70")

    )
    register(
        BedrockLlama("meta.llama3-8b-instruct-v1:0"),
        aliases=("bedrock-llama3-8b-instruct", "bl3-8-i")

    )
    register(
        BedrockLlama("meta.llama3-70b-instruct-v1:0"),
        aliases=("bedrock-llama3-70b-instruct", "bl3-70-i")

    )


class BedrockLlama(llm.Model):
    can_stream: bool = True

    class Options(llm.Options):
        verbose: bool = Field(
            description="Whether to print verbose output from the model", default=False
        )

        max_gen_len: Optional[int] = Field(
            description="The maximum number of tokens to generate before stopping",
            default=2048,
        )

        temperature: Optional[float] = Field(
            description="Temperature",
            default=0.5,
        )

        top_p: Optional[float] = Field(
            description="Top P",
            default=0.9,
        )

        @field_validator("max_gen_len")
        def validate_length(cls, max_gen_len):
            if not (0 < max_gen_len <= 1_000_000):
                raise ValueError("max_gen_len must be in range 1-1,000,000")
            return max_gen_len

    def __init__(self, model_id):
        self.model_id = model_id
        self.default_system_prompt = None

    def build_llama3_instruct_prompt(self, prompt, conversation):
        # <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        # You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
        # {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        system_prompt = None

        if prompt.system:
            system_prompt = prompt.system
        else:
            # Look for a system prompt in the conversation
            if conversation is not None:
                for prev_response in conversation.responses:
                    if prev_response.prompt.system:
                        system_prompt = prev_response.prompt.system

        if system_prompt is None:
            system_prompt = (
                self.default_system_prompt or DEFAULT_LLAMA3_SYSTEM_PROMPT
            )

        if conversation is None:

            return f"""
                    <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
            """

        messages = []
        first = True
        for response in conversation.responses:
            if first:
                messages.append(
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
                )
                first = False

            messages.append(
                f"<|start_header_id|>user<|end_header_id|>\n\n{response.prompt.prompt}<|eot_id|>"
            )
            messages.append(
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{response.text()}<|eot_id|>"
            )

        if not messages:
            messages.append(
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            )
        messages.append(
            f"<|start_header_id|>user<|end_header_id|>\n\n{prompt.prompt}<|eot_id|>"
        )
        messages.append(
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return messages

    def build_llama2_chat_prompt(self, prompt, conversation):
        prompt_bits = []
        # First figure out the system prompt
        system_prompt = None
        if prompt.system:
            system_prompt = prompt.system
        else:
            # Look for a system prompt in the conversation
            if conversation is not None:
                for prev_response in conversation.responses:
                    if prev_response.prompt.system:
                        system_prompt = prev_response.prompt.system

        if system_prompt is None:
            if self.model_id.startswith("meta.llama2"):
                system_prompt = (
                    self.default_system_prompt or DEFAULT_LLAMA2_SYSTEM_PROMPT
                )
            else:
                system_prompt = (
                    self.default_system_prompt or DEFAULT_LLAMA3_SYSTEM_PROMPT
                )

        # Now build the prompt pieces
        first = True
        if conversation is not None:
            for prev_response in conversation.responses:
                prompt_bits.append("<s>[INST] ")
                if first:
                    prompt_bits.append(
                        f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n",
                    )
                first = False
                prompt_bits.append(
                    f"{prev_response.prompt.prompt} [/INST] ",
                )
                prompt_bits.append(
                    f"{prev_response.text()} </s>",
                )

        # Add the latest prompt
        if not prompt_bits:
            # Start with the system prompt
            prompt_bits.append("<s>[INST] ")
            prompt_bits.append(
                f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n",
            )
        else:
            prompt_bits.append("<s>[INST] ")
        prompt_bits.append(f"{prompt.prompt} [/INST] ")
        return prompt_bits

    def execute(self, prompt, stream, response, conversation):
        client = boto3.client('bedrock-runtime')

        if self.model_id.startswith("meta.llama3"):
            prompt_str = "".join(
                self.build_llama3_instruct_prompt(prompt, conversation))
        else:
            prompt_str = "".join(
                self.build_llama2_chat_prompt(prompt, conversation))

        # print(f"Prompt: {prompt_str}")
        prompt_json = {
            "prompt": prompt_str,
            "max_gen_len": prompt.options.max_gen_len,
            "temperature": prompt.options.temperature,
            "top_p": prompt.options.top_p,
        }
        prompt.prompt_json = prompt_json

        if stream:
            bedrock_response = client.invoke_model_with_response_stream(
                modelId=self.model_id, body=json.dumps(prompt_json)
            )
            chunks = bedrock_response.get("body")

            for event in chunks:
                chunk = event.get("chunk")
                if chunk:
                    response = json.loads(chunk.get("bytes").decode())
                    completion = response["generation"]
                    yield completion

        else:
            bedrock_response = client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(prompt_json),
            )
            body = bedrock_response["body"].read()
            response.response_json = json.loads(body)
            completion = response.response_json["generation"]
            yield completion
