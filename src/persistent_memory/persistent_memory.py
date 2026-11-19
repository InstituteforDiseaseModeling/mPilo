from src.multi_language_support.custom_translator import CustomGoogleTranslator

from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage


class PersistentMemory:
    """Persistent memory extraction from chat messages.

    Parameters
    ----------
    llm :
        The LLM to be used for summarizing and merging the important features extracted.
    extractor_prompt : str
        The prompt template string to be used for extracting relevant features from chat history.
    merger_prompt : str
        The prompt template string to be used for merging relevant features from chat history with any previously extracted features.
    """

    def __init__(self, llm, extractor_prompt, merger_prompt):
        self.llm = llm

        self.llm_extractor = self._assemble_llm_chain(
            prompt_string=extractor_prompt, input_placeholder="conversation_input"
        )
        self.llm_merger = self._assemble_llm_chain(prompt_string=merger_prompt, input_placeholder="list_input")

    def _assemble_llm_chain(self, prompt_string, input_placeholder):
        """Assemble the prompt with the LLM provided to create a chain.

        Parameters
        ----------
        prompt_string : str
            The prompt template string to be used by the LLM.
        input_placeholder : str
            The name of the input argument.

        Returns
        -------
        LLMChain
            An LLMChain assembled with the prompt and LLM.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=prompt_string),
                HumanMessagePromptTemplate.from_template("{" + input_placeholder + "}"),
            ]
        )
        llm_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=False)

        return llm_chain

    def _extract_features(self, chat_messages, translate, user_language):
        """Extract important user features from the chat memory.

        Parameters
        ----------
        chat_messages : list
            A list of chat_memory.messages.
        translate : bool
            Whether translation of the chat memory is necessary.
        user_language : str
            The string representing the language of the chat memory.

        Returns
        -------
        str
            A string containing the list of relevant features extracted from the chat memory.
        """
        conversation_string = ""
        for message in chat_messages:
            if type(message) is HumanMessage:
                conversation_string += f"Patient: {message.content}\n"
            if type(message) is AIMessage:
                conversation_string += f"Coach Mpilo: {message.content}\n"

        if translate:
            translate_client = CustomGoogleTranslator()
            t_response = translate_client.translate(
                values=conversation_string, source_language=user_language, target_language="en"
            )
            conversation_string = t_response["translatedText"]

        # Get the name of the input argument
        extractor_response = self.llm_extractor.predict(conversation_input=conversation_string)

        try:
            extracted_features = extractor_response.split("eatures:\n")[1]
        except Exception as ex:
            template = "Error while parsing extracted features: An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            extracted_features = "- No info."

        return extracted_features

    def _merge_features(self, older_features, new_features):
        """Merge two sets of features.

        Parameters
        ----------
        older_features : str
            A string containing the older list of relevant features to be updated.
        new_features : str
            A string containing the newer list of relevant features to update on the older set.

        Returns
        -------
        str
            A string containing the list of relevant features obtained by merging the older and newer sets.
        """
        lists_to_merge = "\nOlder list:\n" + older_features + "\n\nNewer list:\n" + new_features
        # print(lists_to_merge)
        merger_response = self.llm_merger.predict(list_input=lists_to_merge)

        try:
            merged_features = merger_response.split("ist:\n")[1]
        except Exception as ex:
            template = "Error while merging extracted features: An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            merged_features = older_features
        return merged_features

    def extract_and_merge_features(
        self, chat_messages, older_features="- No info.", translate=False, user_language=None
    ):
        """Extract important user features and merge them with any previous features available.

        Parameters
        ----------
        chat_messages : list
            A list of chat_memory.messages.
        older_features : str
            A string containing the older list of relevant features to be updated. (Defaults to "- No info.").
        translate : bool
            Whether translation of the chat memory is necessary. (Defaults to False).
        user_language : str
            The string representing the language of the chat memory. (Defaults to None).

        Returns
        -------
        str
            A string containing a list of relevant features obtained by merging the older and newly extracted sets.
        """
        new_features = self._extract_features(
            chat_messages=chat_messages, translate=translate, user_language=user_language
        )
        merged_features = self._merge_features(older_features=older_features, new_features=new_features)
        return merged_features
