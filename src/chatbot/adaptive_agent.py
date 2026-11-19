from src.config import get_secrets, OPENAI_MODEL
from src.chatbot.agent import (
    CustomOpenAIFunctionAgent,
    CustomReActRunnableAgent,
    TranslateMixin,
    ReActAgent,
    FunctionsAgent,
    STOP_LIST,
)
from src.chatbot.llms import Functionary
from src.chatbot.output_parser import CustomReActOutputParser, FunctionaryOutputParser
from src.chatbot.segmentation import chat_history_prob, new_input_prob, highest_prob_above_threshold, adjust_probs

import warnings
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str, format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.runnable import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_function as format_tool_to_openai_function
from langchain_core.language_models.chat_models import BaseChatModel
from transformers import AutoTokenizer

get_secrets()
INTERNAL_LANGUAGE = "en"
CLASSIFICATION_THRESHOLD = 0.5


class AdaptiveMixin(TranslateMixin):
    """Methods to make an agent adaptive with router for each user segment.

    Parameters
    ----------
    tools : str
        List of LangChain tools made available to the agent.
    classifier_prompt_template : str
        Template for the segment classifier prompt.
    segment_behaviors : dict
        Mapping of additional behavioral advice by identified segment, to be injected into the generative prompt.
    prompt_template_system : str
        Template for the system prompt.
    prompt_template_human : str
        Template for the human prompt.
    max_iterations : int (default is 2)
        Maximum number of functions that can be called before a final output is forced.
    llm : LangChain LLM.
        LLM object instance (default is ChatOpenAI with specific parameters).
    classifier_llm : LangChain LLM.
        LLM object instance for segment classification (default is None, in which case classifier_llm=llm).
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
        classifier_prompt_template,
        segment_behaviors,
        prompt_template_system,
        prompt_template_human,
        max_iterations=2,
        llm=ChatOpenAI(temperature=0, model_name=OPENAI_MODEL, streaming=True),
        classifier_llm=None,
        memory=ConversationBufferMemory(memory_key="chat_history", output_key="output"),
        use_system_prompt=True,
        translate=False,
        external_language=None,
        use_persistent_memory=False,
        trigger_persistent_memory=10,
        retain_k=3,
    ):
        """Init function for all adaptive agents."""
        self.llm = llm
        if "tokenizer" in llm.__dict__:
            self.tokenizer = llm.tokenizer
        else:
            self.tokenizer = None
        self.memory = memory
        self.tools = tools
        self.max_iterations = max_iterations
        self.classifier_prompt_template = classifier_prompt_template
        self.segment_behaviors = segment_behaviors
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

        # Handle LLM used for classification
        if classifier_llm is not None:
            self.classifier_llm = classifier_llm
            self.classifier_chat_model = any(cls in self.llm.__class__.__bases__ for cls in [BaseChatModel, ChatOpenAI])
        else:
            self.classifier_llm = llm
            self.classifier_chat_model = self.chat_model

        # Assemble Segment Classifier LLM chain
        self.classifier_chain = self.create_llm_classifier()

        # Starting probabilities are given by the chat history.
        # If no history then the frequency from Coach Mpilo research is used
        self.segment_probs = chat_history_prob(conversation_buffer=self.memory, llm_classifier=self.classifier_chain)
        print(self.segment_probs)

    def _assemble_classifier_prompt(self):
        """Create dynamic prompt from template for the LLM classifier.

        Returns
        -------
        langchain.prompts.PromptTemplate
            Prompt with known details filled in.
        """
        if self.classifier_chat_model:
            system_message_prompt = SystemMessagePromptTemplate.from_template(self.classifier_prompt_template)
            prompt = ChatPromptTemplate.from_messages([system_message_prompt])
        else:
            # Convert the Chat prompt into a simple prompt, using the tokenizer's apply_chat_template method to apply the appropriate template for each model.
            messages = [
                {
                    "role": "user",
                    "content": self.classifier_prompt_template,
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

        return prompt

    def create_llm_classifier(self):
        """Create a chain that will evaluate user input and return the probability of the user being assigned to each segment.

        Returns
        -------
        RunnableSequence
            A Langchain RunnableSequence chain that returns segment classification probability.
        """
        assembled_seg_prompt = self._assemble_classifier_prompt()
        classifier_chain = assembled_seg_prompt | self.classifier_llm
        return classifier_chain

    def _generate_segment_prompt(self, info):
        """Generate a prompt template based on the identified segment.

        Parameters
        ----------
        info : dict
            Dictionary containing LangChain chain inputs and intermediate variables.

        Returns
        -------
        ChatPromptTemplate
            A prompt with segment specific instructions to be used in the agent chain.
        """
        # Priori segment assignation probs
        priori_probs = self.segment_probs

        # Only update probs if this is the first time the chain has been run on an input.
        if info["agent_scratchpad"] in [[], ""]:

            # Posteriori probs based on new input, given past history
            posteriori_probs = new_input_prob(
                conversation_buffer=self.memory, new_input=info["input"], llm_classifier=self.classifier_chain
            )

            posteriori_probs.rename(columns={"prob": "new_prob", "evidence": "new_evidence"}, inplace=True)
            new_probs = adjust_probs(priori_probs, posteriori_probs)
            self.segment_probs = new_probs
            # print(self.segment_probs)

        segment = highest_prob_above_threshold(self.segment_probs, "Opaque", CLASSIFICATION_THRESHOLD)

        if segment.lower() != "opaque":
            seg_start = "\nBear in mind the following additional advice based on the assigned segment of the human:"
            segment_instruction = seg_start + self.segment_behaviors[segment]
            final_sys_prompt = self.prompt_template_system + segment_instruction
        else:
            final_sys_prompt = self.prompt_template_system

        self.prompt_template_system = final_sys_prompt

        # Assemble prompt
        final_prompt = self._assemble_prompt()
        return final_prompt

    def to_chain(self, format_func, llm_with_stop):
        """Generate a chain primitive with segment-based prompt.

        Parameters
        ----------
        format_func : function
            Langchain provided function to format intermediate steps appropriately for agent type.
        llm_with_stop : LangChain LLM
            Langchain LLM with stopwords and functions added (bound).

        Returns
        -------
        RunnableSequence
            A Langchain RunnableSequence with segment-specific instructions.
        """
        agent_chain = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_func(x["intermediate_steps"]),
                "chat_history": lambda x: x["chat_history"],
            }
            | RunnableLambda(self.persistent_memory)
            | RunnableLambda(self._translate_input)
            | RunnableLambda(self._generate_segment_prompt)
            | llm_with_stop
        )
        return agent_chain


class AdaptiveReActAgent(ReActAgent, AdaptiveMixin):
    """Agent that follows the ReAct framework."""

    def __init__(self, *args, **kwargs):
        """Initialize Agent."""
        AdaptiveMixin.__init__(self, *args, **kwargs)

    def to_executor(self, observation_prefix="Observation: ", llm_prefix="Thought: "):
        """Assemble Agent in LCEL and return AgentExecutor instance.

        Returns
        -------
        langchain.agents.AgentExecutor
        """
        stop = STOP_LIST
        llm_with_stop = self.llm.bind(stop=STOP_LIST)
        agent_chain = (
            self.to_chain(format_log_to_str, llm_with_stop)
            | CustomReActOutputParser()
            | RunnableLambda(self._translate_output)
        )

        agent_executor = AgentExecutor(
            agent=CustomReActRunnableAgent(
                runnable=agent_chain, observation_prefix=observation_prefix, llm_prefix=llm_prefix, _stop=stop
            ),
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            early_stopping_method="generate",
            handle_parsing_errors=True,
            max_iterations=1,
        )

        return agent_executor

    def _handle_error(error) -> str:
        return str(error)[:50]


class AdaptiveFunctionsAgent(FunctionsAgent, AdaptiveMixin):
    """Agent that follows the Functions framework."""

    def __init__(self, *args, **kwargs):
        """Initialize Agent."""
        AdaptiveMixin.__init__(self, *args, **kwargs)

    def to_executor(self, observation_prefix="Observation: ", llm_prefix="Thought: "):
        """Assemble Agent in LCEL and return AgentExecutor instance.

        Parameters
        ----------
        observation_prefix : str
            String denoting the prefix used for observations (i.e. user input).
        llm_prefix : str
            String denoting the prefix used for agent thoughts.

        Returns
        -------
        langchain.agents.AgentExecutor
        """
        llm_with_stop = self.llm.bind(
            stop=STOP_LIST,
            functions=[format_tool_to_openai_function(t) for t in self.tools],
        )
        parser = FunctionaryOutputParser() if self.llm.__class__ == Functionary else OpenAIFunctionsAgentOutputParser()
        runnable = (
            self.to_chain(format_to_openai_function_messages, llm_with_stop)
            | parser
            | RunnableLambda(self._translate_output)
        )

        agent_executor = AgentExecutor(
            agent=CustomOpenAIFunctionAgent(
                runnable=runnable, observation_prefix=observation_prefix, llm_prefix=llm_prefix, _stop=STOP_LIST
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
