from src import config
from awq import AutoAWQForCausalLM
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from pydantic import ConfigDict
import subprocess
import atexit
from pydantic.v1 import root_validator
import time

import json
import boto3
from typing import Any, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM


class Llama2(HuggingFacePipeline):
    """Llama-2-13B-Chat-AWQ LLM."""

    generation_params: dict = {
        "do_sample": False,
        "max_new_tokens": 1000,
    }
    tokenizer: Any
    model: Any = None

    @root_validator(pre=False)
    def initialize_model(cls, values):
        """Initialize LLM."""
        values["model"] = AutoAWQForCausalLM.from_quantized(
            "TheBloke/Llama-2-13B-chat-AWQ", fuse_layers=True, trust_remote_code=False, safetensors=True
        )
        tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-13B-chat-AWQ", use_fast=True)
        values["pipeline"] = pipeline(
            "text-generation", model=values["model"].model, tokenizer=tokenizer, **values["generation_params"]
        )
        values["tokenizer"] = tokenizer
        return values


class Llama2_70B(LLM):
    """TheBloke/Llama-2-70B-chat-AWQ AWS Bedrock."""

    client: Any
    tokenizer: Any

    @property
    def _llm_type(self) -> str:
        return "Bedrock"

    @root_validator(pre=False)
    def initialize_model(cls, values):
        """Initialize LLM."""
        tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-70B-chat-AWQ", use_fast=True)
        values["client"] = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
        values["tokenizer"] = tokenizer
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        model_id = "meta.llama2-70b-chat-v1"
        body = {
            "prompt": prompt,
            "temperature": 0,
            "max_gen_len": 1500,
        }
        try:
            response = self.client.invoke_model_with_response_stream(
                modelId=model_id,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json",
            )
            stream = response["body"]
            generation = ""
            stopped = False
            if stream:
                for event in stream:
                    chunk = event.get("chunk")
                    if chunk:
                        text = json.loads(chunk.get("bytes").decode()).get("generation")
                        if text:
                            generation += text
                    if stop:
                        # Check for each stop string and break output if found.
                        for stop_str in stop:
                            pos = generation.rfind(stop_str)
                            if pos != -1:
                                generation = generation[:pos]
                                stopped = True
                                break
                            else:
                                pass
                        if stopped:
                            break
            return generation.strip()

        except Exception as e:
            raise ValueError(f"Error raised by bedrock service: {e}")


class Meditron(HuggingFacePipeline):
    """Meditron-7B-AWQ LLM."""

    generation_params: dict = {
        "do_sample": False,
        "max_new_tokens": 1000,
    }
    tokenizer: Any
    model: Any = None

    @root_validator(pre=False)
    def initialize_model(cls, values):
        """Initialize LLM."""
        values["model"] = AutoAWQForCausalLM.from_quantized(
            "TheBloke/meditron-7B-AWQ", fuse_layers=True, trust_remote_code=False, safetensors=True
        )
        tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b", use_fast=True)
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        values["pipeline"] = pipeline(
            "text-generation", model=values["model"].model, tokenizer=tokenizer, **values["generation_params"]
        )
        values["tokenizer"] = tokenizer

        return values


class Mixtral(HuggingFacePipeline):
    """TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ LLM."""

    model_config = ConfigDict(extra="ignore")
    generation_params: dict
    tokenizer: Any

    def __init__(
        self,
        generation_params: dict = {
            "do_sample": False,
            "max_new_tokens": 1000,
        },
    ) -> None:
        """Initialize LLM."""
        model_name_or_path = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto", trust_remote_code=False, revision="gptq-3bit-128g-actorder_True"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **generation_params)
        super().__init__(pipeline=pipe, generation_params=generation_params, tokenizer=tokenizer)
        self.tokenizer = tokenizer


class MistralFunctionCalling(HuggingFacePipeline):
    """Trelis/Mistral-7B-Instruct-v0.2-function-calling-v3."""

    model_config = ConfigDict(extra="ignore")
    generation_params: dict
    tokenizer: Any

    def __init__(
        self,
        generation_params: dict = {
            "do_sample": False,
            "max_new_tokens": 1000,
        },
    ) -> None:
        """Initialize LLM."""
        model_name_or_path = "glaiveai/glaive-function-calling-v1"
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            trust_remote_code=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=True)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **generation_params)
        super().__init__(pipeline=pipe, generation_params=generation_params, tokenizer=tokenizer)
        self.tokenizer = tokenizer


class Functionary(ChatOpenAI):
    """Functionary LLM."""

    model_name: str = "meetkai/functionary-7b-v2.1"
    openai_api_base: str = "http://localhost:8000/v1"
    openai_api_key: str = "functionary"
    temperature: float = 0
    streaming: bool = True
    server_process: subprocess.Popen = None

    @root_validator(pre=False)
    def initialize_server(cls, values):
        """Alternative init method for a Pydantic Model."""
        print("Wait for Uvicorn server to start..")
        awq_model_folder = f"{values['model_name'].split('meetkai/')[1]}-awq"
        # Download model to the models folder if it doesn't exist yet.
        subprocess.run(
            [
                "aws",
                "s3",
                "sync",
                f"{config.MODELS_S3_URI}/{awq_model_folder}",
                f"../../models/{awq_model_folder}",
                "--delete",
            ]
        )
        # Start the server
        values["server_process"] = subprocess.Popen(
            ["python", "../../functionary/server.py", "--model", f"../../models/{awq_model_folder}"]
        )

        # Allow some time for server to start
        time.sleep(10)

        atexit.register(values["server_process"].terminate)

        return values

    def __del__(self):
        """Ensure the server closes to free up GPU VRAM when the class instance is deleted."""
        if self.server_process:
            self.server_process.terminate()  # Ensure the subprocess is terminated when the class instance is deleted
