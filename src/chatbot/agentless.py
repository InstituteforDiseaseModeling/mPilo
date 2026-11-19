from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import MessagesPlaceholder
from transformers import AutoTokenizer
from operator import itemgetter
from src.chatbot.prompt import COACH_MPILO_PROMPT_SYSTEM
import warnings
import time

AGENTLESS_PROMPT_HUMAN = """Previous conversation history: {chat_history}

New input: {input}
"""

AGENTLESS_PROMPT_HUMAN_CHAT = "{input}"


class DictOutputParser(StrOutputParser):
    """Return output as dict with the string output as a key, similar to Agent output."""

    def parse(self, text):
        """Parse output text."""
        return {"output": text}


class Agentless:
    """LLM without retrieval-augmentation to establish baseline performance on ODQA tasks.

    Parameters
    ----------
    llm : LangChain LLM
        LLM object instance (default is ChatOpenAI with specific parameters).
    memory: ConversationBufferMemory, optional
        ConversationBufferMemory instance (default is initialized with specific memory_key and output_key)
    tokenizer_chat_template: str, optional
        String containing the chat template for the tokenizer
    prompt_template_human : str, optional
        Template for the human prompt.
    prompt_template_system : str, optional
        Template for the system prompt.
    use_system_prompt: bool, optional
        Whether the system prompt should be used
    """

    def __init__(
        self,
        llm,
        tokenizer_chat_template=None,
        prompt_template_human=None,
        prompt_template_system=COACH_MPILO_PROMPT_SYSTEM,
        use_system_prompt=True,
        custom_prompt_template=None,
        memory=ConversationBufferMemory(memory_key="chat_history", output_key="output"),
    ):
        self.tokenizer_chat_template = tokenizer_chat_template
        self.llm = llm
        self.memory = memory
        if "tokenizer" in self.llm.__dict__:
            self.tokenizer = self.llm.tokenizer
        else:
            self.tokenizer = None
        self.chat_model = any(cls in self.llm.__class__.__bases__ for cls in [BaseChatModel, ChatOpenAI])
        self.use_system_prompt = use_system_prompt
        self.prompt_template_system = prompt_template_system
        if prompt_template_human:
            self.prompt_template_human = prompt_template_human
        elif self.chat_model:
            self.prompt_template_human = AGENTLESS_PROMPT_HUMAN_CHAT
        else:
            self.prompt_template_human = AGENTLESS_PROMPT_HUMAN
        self.prompt = (
            PromptTemplate.from_template(custom_prompt_template)
            if custom_prompt_template is not None
            else self._assemble_prompt()
        )
        self.chain = self.create_chain()

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
            prompt = ChatPromptTemplate.from_messages(
                [system_message_prompt, MessagesPlaceholder(variable_name="chat_history"), human_message_prompt]
            )
            # Ensure memory returns ChatMessages, rather than strings
            self.memory.return_messages = True
        elif self.prompt_template_system and self.prompt_template_human:
            self.memory.return_messages = False
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

        return prompt

    def create_chain(self):
        """
        Create LCEL chain w/ memory for processing queries.

        Returns langchain_core.runnables.base.RunnableSequence.
        """
        llm_chain = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("chat_history")
            )
            | self.prompt
            | self.llm.bind(stop=["\n\n"])
            | DictOutputParser()
        )
        return llm_chain

    def to_executor(self):
        """Simulate agent's to_executor method for cross-compatibility.

        Returns
        -------
            Langchain Chain
        """
        return self.chain

    def invoke(self, input_dict, print_time=False):
        """
        Pass input to the chain, behaving like an agent executor.

        print_time: bool
            Whether to print the elapsed time for LLM response.
        input_dict: dict
            Input dictionary, e.g. {"input": "what is the treatment for 14yo males?"}

        Returns
        -------
        str
            Response from Agentless
        """
        start = time.time()
        result = self.chain.invoke(input_dict)
        end = time.time()
        self.memory.save_context(input_dict, {"output": result})
        if print_time:
            print(f"Elapsed Time: {end - start}")
            print("------------------------------")
        return result.strip()
