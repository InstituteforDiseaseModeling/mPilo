import gradio as gr
from langchain.schema.agent import AgentAction
from langchain_core.agents import AgentActionMessageLog
from src.chatbot.adaptive_agent import AdaptiveReActAgent, AdaptiveFunctionsAgent
from langchain.schema import AIMessage, HumanMessage
from src.config import get_secrets
from src.chatbot.segmentation import chat_history_prob

get_secrets()

TITLE = "Mpilo Demo"
CHATBOT_HEIGHT = 800
CHATBOT_THEME = gr.themes.Default(neutral_hue="slate", primary_hue="green", secondary_hue="blue")


class FrontendDemo:
    """Frontent demo for chatbot."""

    def __init__(self, agent, show_sources=False):
        """Initialize frontend.

        Parameters
        ----------
        agent : AgentExecutor
            Langchain agent object.
        """
        self.agent = agent
        self.agent_executor = agent.to_executor()
        self.show_sources = show_sources
        self.theme = CHATBOT_THEME
        self.gui = self._build_gui()
        self._clear()

    def _build_gui(self):
        """Assemble the Gradio GUI interface.

        Returns
        -------
        Gradio GUI
        """
        with gr.Blocks(theme=self.theme) as gui:
            chatbot = gr.Chatbot(label=TITLE, height=CHATBOT_HEIGHT)
            msg = gr.Textbox(show_label=False, placeholder="Message chatbot..")
            clear = gr.Button("Clear")

            msg.submit(self._user, [msg, chatbot], [msg, chatbot], queue=False).then(self._call, chatbot, chatbot)
            clear.click(self._clear, None, chatbot, queue=False)
            return gui

    def launch(self, *args, **kwargs):
        """Launch the GUI interface."""
        return self.gui.launch(*args, **kwargs)

    def _clear(self):
        """Clear the agent memory."""
        if type(self.agent) in [AdaptiveReActAgent, AdaptiveFunctionsAgent]:
            self.agent.segment_probs = chat_history_prob(
                conversation_buffer=self.agent_executor.memory, llm_classifier=self.agent.classifier_chain
            )

    def _call(self, history):
        """Call the chatbot agent and handle interaction with Gradio history.

        Parameters
        ----------
        history : list
            Nested list, consisting of 2-item lists [user_message, ai_message] in conversation order. This is what gets displayed in Gradio.

        Yields
        ------
        list
            Updated history list.
        """
        return_text = ""
        agent_text = ""
        references = ""
        history_langchain_format = []
        for human, ai in history[:-1]:
            if human:
                history_langchain_format.append(HumanMessage(content=human))
            if ai:
                ai = ai.split("\n**Retrieval Sources**\n")[0]
                history_langchain_format.append(AIMessage(content=ai))
        if self.agent_executor.memory.chat_memory:
            self.agent_executor.memory.chat_memory.messages = history_langchain_format
        for chunk in self.agent_executor.stream({"input": history[-1][0]}):
            if "output" in chunk.keys():
                agent_text += chunk["output"]
            if "intermediate_steps" in chunk.keys():
                if (len(chunk["intermediate_steps"]) > 0) and self.show_sources:
                    references += self._documents_retrieved(chunk["intermediate_steps"])
            return_text = agent_text + "\n" + references
            history[-1][1] = return_text
            yield history

    def _documents_retrieved(self, intermediate_steps):
        """Generate markdown representation of documents retrieved for display in GUI.

        Parameters
        ----------
        intermediate_steps : list
            List of agent intermediate steps provided by LangChain.

        Returns
        -------
        str
            String markdown representation.
        """
        retrieval_actions = [
            action for action in intermediate_steps if (type(action[0]) in [AgentAction, AgentActionMessageLog])
        ]
        retrieval_outputs = [output[1] for output in retrieval_actions]
        print(retrieval_outputs)
        return "**Retrieval Sources**\n```\n" + "\n\n".join(retrieval_outputs) + "\n```"

    def _user(self, user_message, history):
        """Update the user message in the GUI while waiting for the chatbot response.

        Parameters
        ----------
        user_message : str
            Most recent user message.
        history : list
            Nested list, consisting of 2-item lists [user_message, ai_message] in conversation order. This is what gets displayed in Gradio.

        Returns
        -------
        tuple
            Empty string to reset the message box, along with updated history.
        """
        return "", history + [[user_message, None]]


if __name__ == "__main__":
    demo = ChatbotDemo(agent)
    demo.launch()
