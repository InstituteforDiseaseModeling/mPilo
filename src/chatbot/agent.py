from src.config import get_secrets, OPENAI_MODEL
from src.chatbot.llms import Functionary
from src.chatbot.output_parser import CustomReActOutputParser, FunctionaryOutputParser
from src.multi_language_support.custom_translator import CustomGoogleTranslator

from langchain.agents import AgentExecutor
from langchain.agents.agent import RunnableAgent
from langchain.agents.format_scratchpad import format_log_to_str, format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.runnable import RunnableLambda
from langchain.tools.render import render_text_description
from langchain_core.utils.function_calling import convert_to_openai_function as format_tool_to_openai_function
from langchain_core.agents import AgentFinish
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.base import RunnableBinding
from langchain_openai import ChatOpenAI

from pydantic import ConfigDict
from transformers import AutoTokenizer
from typing import List

import subprocess
import warnings

get_secrets()
STOP_LIST = ["\nObservation", "\nNew input", "</s>"]
INTERNAL_LANGUAGE = "en"


class MemoryMixin:
    """Memory handling for agent."""

    def inject_user_details(self, user_name, user_details, silent=True):
        """Add user details to the agent.

        Parameters
        ----------
        user_name : str
            A string with the user name.
        user_details : str
            A string containing the list of user details extracted from previous chat interactions.

        Returns
        -------
        bool
            True when complete.
        """
        self.user_name = user_name

        if user_details is not None or user_details != "":
            # Retain original prompt to avoid stacking of details
            if "orig_prompt_template" not in dir(self):
                if not silent:
                    print("Storing original prompt template")
                self.orig_prompt_template = self.prompt_template_system
            details_start = "\nBear in mind the following information about the user when answering:\n"
            details_instruction = details_start + user_details
            self.prompt_template_system = self.orig_prompt_template + details_instruction
            # Update prompt
            self.prompt = self._assemble_prompt()
            if not silent:
                print("User details added to prompt")
        return True

    def persistent_memory(self, info):
        """Check chat history length and triggers persistent memory extraction if needed.

        Parameters
        ----------
        info : dict
            Dictionary containing LangChain chain inputs and intermediate variables.

        Returns
        -------
        dict
            Pass along the info from input.
        """
        # Only check at first agent iteration
        if info["agent_scratchpad"] in [[], ""] and self.use_persistent_memory:
            # Count interactions
            num_ai_responses = info["chat_history"].count("\nAI:")
            if num_ai_responses >= self.trigger_persistent_memory:
                # Call MemoryManager to update user persistent memories
                print("Calling MemoryManager...")
                bash_command = "python persistent_memory_runnable.py {v1} {v2} {v3} {v4}"
                bash_command = bash_command.format(
                    v1=self.user_name, v2=self.translate, v3=self.external_language, v4=self.retain_k
                )
                # Call to run on the background
                subprocess.Popen(bash_command.split(), cwd="../../src/persistent_memory")

        return info


class TranslateMixin:
    """Google Translate Layers for agents.

    Parameters
    ----------
    translate : bool, optional
        Whether to use the translation layers (default is False). If True, input will be translated into the LLM language and the LLM output will be translated into the detected input language.
    external_language : str, optional
        The language the user is using to communicate with the LLM (default is None). If None, external language will be detected automatically by the Google Translate API.
    """

    def __init__(
        self,
        translate=False,
        external_language=None,
    ):
        # Translation parameters
        self.internal_language = INTERNAL_LANGUAGE
        self.external_language = external_language
        self.fixed_external_language = external_language
        if translate:
            self.translate_client = CustomGoogleTranslator()
        if self.internal_language == self.external_language:
            self.translate = False
        else:
            self.translate = translate

    def _translate_input(self, info):
        """Define translation layer for user input."""
        if self.translate:
            # Input translation has to be applied every iteration of the chain
            translated_input = self.translate_client.translate(
                values=info["input"], source_language=self.external_language, target_language=self.internal_language
            )
            if self.external_language is None:
                self.external_language = translated_input["detectedSourceLanguage"]
            info["input"] = translated_input["translatedText"]
        return info

    def _translate_output(self, info):
        """Define translation layer for Agent output."""
        if self.translate:
            # Only translate output if this is the last agent iteration.
            if type(info) == AgentFinish:
                translated_output = self.translate_client.translate(
                    values=info.return_values["output"],
                    source_language=self.internal_language,
                    target_language=self.external_language,
                )
                # Reset detected language for next input
                self.external_language = self.fixed_external_language
                info.return_values["output"] = translated_output["translatedText"]

        return info


class ReActAgent(TranslateMixin, MemoryMixin):
    """Agent that follows the ReAct framework.

    Parameters
    ----------
    tools : str
        List of LangChain tools made available to the agent.
    llm : LangChain LLM
        LLM object instance (default is ChatOpenAI with specific parameters).
    memory : ConversationBufferMemory, optional
        ConversationBufferMemory instance (default is initialized with specific memory_key and output_key).
    prompt_template_instruct : str, optional
        Template for the full instruction prompt.
    prompt_template_system : str, optional
        Template for the system prompt.
    prompt_template_human : str, optional
        Template for the human prompt.
    use_system_prompt : bool, optional
        Whether to use the system prompt (default is True). Useful for instruct models such as Mistral that don't accept a system prompt in their template.
    translate : bool, optional
        Whether to use the translation layers (default is False). If True, input will be translated into the LLM language and the LLM output will be translated into the detected input language.
        translate : bool, optional
        Whether to use the translation layers (default is False). If True, input will be translated into the LLM language and the LLM output will be translated into the detected input language.
    external_language : str, optional
        The language the user is using to communicate with the LLM (default is None). If None, external language will be detected automatically by the Google Translate API.
    use_persistent_memory : bool
        If the PersistentMemory module is to be used (default is False).
    trigger_persistent_memory : int
        If use_persistent_memory, will trigger a persistent memory extraction when the number of chat interactions is greater than trigger_persistent_memory (default is 10).
    retain_k : int
        If use_persistent_memory, the last k interactions are kept in the short term memories after persistent memories are extracted from the chat history (default is 3).
    """

    def __init__(
        self,
        tools,
        llm=ChatOpenAI(temperature=0, model_name=OPENAI_MODEL, streaming=True),
        memory=ConversationBufferWindowMemory(memory_key="chat_history", output_key="output", k=8),
        prompt_template_instruct=None,
        prompt_template_system=None,
        prompt_template_human=None,
        use_system_prompt=True,
        translate=False,
        external_language=None,
        use_persistent_memory=False,
        trigger_persistent_memory=10,
        retain_k=3,
    ):
        """Initialize Agent."""
        self.llm = llm
        if "tokenizer" in llm.__dict__:
            self.tokenizer = llm.tokenizer
        else:
            self.tokenizer = None
        self.memory = memory
        self.tools = tools
        self.prompt_template_instruct = prompt_template_instruct
        self.prompt_template_system = prompt_template_system
        self.prompt_template_human = prompt_template_human
        self.use_system_prompt = use_system_prompt

        # Initialize Translation parameters
        TranslateMixin.__init__(self, translate=translate, external_language=external_language)

        # Initialize Persistent Memory parameters
        self.use_persistent_memory = use_persistent_memory
        self.trigger_persistent_memory = trigger_persistent_memory
        self.retain_k = retain_k

        if self.prompt_template_system:
            assert self.prompt_template_human, "If in chat mode, both a human and system prompt must be passed."
        if self.prompt_template_instruct:
            assert (not self.prompt_template_system) and (
                not prompt_template_human
            ), "Either an instruction or chat prompts should be passed, not both."

        # Identify whether model is a chat model
        self.chat_model = any(cls in self.llm.__class__.__bases__ for cls in [BaseChatModel, ChatOpenAI])

        # Assemble prompt
        self.prompt = self._assemble_prompt()

    def _assemble_prompt(self):
        """Create dynamic prompt from template.

        Returns
        -------
        langchain.prompts.PromptTemplate
            Prompt with known details filled in.
        """
        if self.chat_model:
            # print("Is chat model")
            system_message_prompt = SystemMessagePromptTemplate.from_template(self.prompt_template_system)
            human_message_prompt = HumanMessagePromptTemplate.from_template(self.prompt_template_human)
            prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        elif self.prompt_template_system and self.prompt_template_human:
            # Convert the Chat prompt into a simple prompt, using the tokenizer's apply_chat_template method to apply the appropriate template for each model.
            if self.use_system_prompt:
                messages = [
                    {
                        "role": "system",
                        "content": self.prompt_template_system,
                    },
                    {
                        "role": "user",
                        "content": self.prompt_template_human,
                    },
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": self.prompt_template_system + "\n" + self.prompt_template_human,
                    },
                ]

            if self.tokenizer:
                prompt_template = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                warnings.warn("Non-chat models require a tokenizer to identify their chat template")
                tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
                prompt_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            prompt = PromptTemplate.from_template(prompt_template)
        else:
            if self.prompt_template_instruct:
                prompt_template = self.prompt_template_instruct
                prompt = PromptTemplate.from_template(prompt_template)
            else:
                raise Exception("A prompt must be provided to the agent through one of the prompt_template parameters.")

        prompt = prompt.partial(
            tools=render_text_description(self.tools),
            tool_names=", ".join([t.name for t in self.tools]),
        )
        return prompt

    def to_executor(self, observation_prefix="Observation: ", llm_prefix="Thought: "):
        """Assemble Agent in LCEL and return AgentExecutor instance.

        Returns
        -------
        langchain.agents.AgentExecutor
        """
        stop = STOP_LIST
        llm_with_stop = self.llm.bind(stop=STOP_LIST)

        runnable = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
                "chat_history": lambda x: x["chat_history"],
            }
            | RunnableLambda(self.persistent_memory)
            | RunnableLambda(self._translate_input)
            | self.prompt
            | llm_with_stop
            | CustomReActOutputParser()
            | RunnableLambda(self._translate_output)
        )
        agent_executor = AgentExecutor(
            agent=CustomReActRunnableAgent(
                runnable=runnable, observation_prefix=observation_prefix, llm_prefix=llm_prefix, _stop=stop
            ),
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            early_stopping_method="generate",
            handle_parsing_errors="Check your output and make sure it conforms!",  # self._handle_error for custom error handling
            # handle_parsing_errors=True,
            max_iterations=3,
        )

        return agent_executor


class FunctionsAgent(TranslateMixin, MemoryMixin):
    """Agent that uses OpenAI Functions, or similar.

    Parameters
    ----------
    tools : str
        List of LangChain tools made available to the agent.
    prompt_template_system : str
        Template for the system prompt.
    prompt_template_human : str
        Template for the human prompt.
    max_iterations : int (default is 2)
        Maximum number of functions that can be called before a final output is forced.
    llm : LangChain LLM
        LLM object instance (default is ChatOpenAI with specific parameters).
    memory : ConversationBufferMemory, optional
        ConversationBufferMemory instance (default is initialized with specific memory_key and output_key).
    use_system_prompt : bool, optional
        Whether to use the system prompt (default is True). Useful for instruct models such as Mistral that don't accept a system prompt in their template.
    translate : bool, optional
        Whether to use the translation layers (default is False). If True, input will be translated into the LLM language and the LLM output will be translated into the detected input language.
    external_language : str, optional
        The language the user is using to communicate with the LLM (default is None). If None, external language will be detected automatically by the Google Translate API.
    use_persistent_memory : bool
        If the PersistentMemory module is to be used (default is False).
    trigger_persistent_memory : int
        If use_persistent_memory, will trigger a persistent memory extraction when the number of chat interactions is greater than trigger_persistent_memory (default is 10).
    retain_k : int
        If use_persistent_memory, the last k interactions are kept in the short term memories after persistent memories are extracted from the chat history (default is 3).
    """

    def __init__(
        self,
        tools,
        prompt_template_system,
        prompt_template_human,
        max_iterations=2,
        llm=ChatOpenAI(temperature=0, model_name=OPENAI_MODEL, streaming=True),
        memory=ConversationBufferMemory(memory_key="chat_history", output_key="output"),
        use_system_prompt=True,
        translate=False,
        external_language=None,
        use_persistent_memory=False,
        trigger_persistent_memory=10,
        retain_k=3,
    ):
        """Initialize Agent."""
        self.llm = llm
        if "tokenizer" in llm.__dict__:
            self.tokenizer = llm.tokenizer
        else:
            self.tokenizer = None
        self.memory = memory
        self.tools = tools
        self.max_iterations = max_iterations
        self.prompt_template_system = prompt_template_system
        self.prompt_template_human = prompt_template_human
        self.use_system_prompt = use_system_prompt

        # Initialize Translation parameters
        TranslateMixin.__init__(self, translate=translate, external_language=external_language)

        # Initialize Persistent Memory parameters
        self.use_persistent_memory = use_persistent_memory
        self.trigger_persistent_memory = trigger_persistent_memory
        self.retain_k = retain_k

        # Identify whether model is a chat model
        self.chat_model = any(cls in self.llm.__class__.__bases__ for cls in [BaseChatModel, ChatOpenAI])

        # Assemble prompt
        self.prompt = self._assemble_prompt()

    def _assemble_prompt(self):
        """Create dynamic prompt from template.

        Returns
        -------
        langchain.prompts.PromptTemplate
            Prompt with known details filled in.
        """
        if self.chat_model:
            system_message_prompt = SystemMessagePromptTemplate.from_template(self.prompt_template_system)
            human_message_prompt = HumanMessagePromptTemplate.from_template(self.prompt_template_human)
            prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        else:
            # Convert the Chat prompt into a simple prompt, using the tokenizer's apply_chat_template method to apply the appropriate template for each model.
            if self.use_system_prompt:
                messages = [
                    {
                        "role": "system",
                        "content": self.prompt_template_system,
                    },
                    {
                        "role": "user",
                        "content": self.prompt_template_human,
                    },
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": self.prompt_template_system + "\n" + self.prompt_template_human,
                    },
                ]

            if self.tokenizer:
                prompt_template = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                raise Exception("Non-chat models require a tokenizer to identify their chat template")

            prompt = PromptTemplate.from_template(prompt_template)

        prompt = prompt.partial(
            tools=render_text_description(self.tools),
            tool_names=", ".join([t.name for t in self.tools]),
        )
        return prompt

    def to_executor(self, observation_prefix="Observation: ", llm_prefix="Thought: "):
        """Assemble Agent in LCEL and return AgentExecutor instance.

        Returns
        -------
        langchain.agents.AgentExecutor
        """
        stop = STOP_LIST
        llm_with_stop = self.llm.bind(
            stop=stop,
            functions=[format_tool_to_openai_function(t) for t in self.tools],
        )
        parser = FunctionaryOutputParser() if self.llm.__class__ == Functionary else OpenAIFunctionsAgentOutputParser()

        runnable = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
                "chat_history": lambda x: x["chat_history"],
            }
            | RunnableLambda(self.persistent_memory)
            | RunnableLambda(self._translate_input)
            | self.prompt
            | llm_with_stop
            | parser
            | RunnableLambda(self._translate_output)
        )

        agent_executor = AgentExecutor(
            agent=CustomOpenAIFunctionAgent(
                runnable=runnable, observation_prefix=observation_prefix, llm_prefix=llm_prefix, _stop=stop
            ),
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            early_stopping_method="generate",
            handle_parsing_errors=True,  # self._handle_error for custom error handling
            max_iterations=self.max_iterations,
        )

        return agent_executor


class CustomOpenAIFunctionAgent(RunnableAgent):
    """Customize the Agent class for Runnables to implement early stopping."""

    config = ConfigDict(extra="allow")
    observation_prefix: str
    llm_prefix: str
    _stop: List = []

    def return_stopped_response(
        self,
        early_stopping_method,
        intermediate_steps,
        **kwargs,
    ):
        """Return response when agent has been stopped due to max iterations."""
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish({"output": "Agent stopped due to iteration limit or time limit."}, "")
        elif early_stopping_method == "generate":
            # Generate does one final forward pass
            messages = format_to_openai_function_messages(intermediate_steps)
            # Attempt to find the middle runnable and extract the functions to remove them
            binding = [item for item in self.runnable.middle if type(item) == RunnableBinding][0]
            # save the functions
            functions = binding.kwargs["functions"]
            # Remove functions from the agent temporarily
            binding.kwargs.pop("functions")
            new_inputs = {"agent_scratchpad": messages, "intermediate_steps": intermediate_steps}
            full_inputs = {**kwargs, **new_inputs}
            parsed_output = self.runnable.invoke(full_inputs)
            # Replace the functions
            binding.kwargs["functions"] = functions
            if isinstance(parsed_output, AgentFinish):
                # If we can extract, we send the correct stuff
                return parsed_output
            else:
                # Give up, return a friendly message
                return AgentFinish(
                    {"output": "I'm sorry, I had trouble processing that. Could you ask in a different way?"}, ""
                )
        else:
            raise ValueError(
                "early_stopping_method should be one of `force` or `generate`, " f"got {early_stopping_method}"
            )


class CustomReActRunnableAgent(RunnableAgent):
    """Customize the Agent class for Runnables to implement early stopping."""

    config = ConfigDict(extra="allow")
    observation_prefix: str
    llm_prefix: str
    _stop: List = []

    def return_stopped_response(
        self,
        early_stopping_method,
        intermediate_steps,
        **kwargs,
    ):
        """Return response when agent has been stopped due to max iterations."""
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish({"output": "Agent stopped due to iteration limit or time limit."}, "")
        elif early_stopping_method == "generate":
            # Generate does one final forward pass
            intermediate_steps += [
                (intermediate_steps[-1][0], "I now need to return a final answer based on the previous steps:")
            ]
            # Restrict to final 1 list items
            intermediate_steps = intermediate_steps[-1:]
            new_inputs = {"intermediate_steps": intermediate_steps}
            full_inputs = {**kwargs, **new_inputs}
            parsed_output = self.runnable.invoke(full_inputs)
            if isinstance(parsed_output, AgentFinish):
                # If we can extract, we send the correct stuff
                return parsed_output
            else:
                # Give up, return a friendly message
                return AgentFinish(
                    {"output": "I'm sorry, I had trouble processing that. Could you ask in a different way?"}, ""
                )
        else:
            raise ValueError(
                "early_stopping_method should be one of `force` or `generate`, " f"got {early_stopping_method}"
            )
