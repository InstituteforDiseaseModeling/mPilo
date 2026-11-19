from src.chatbot.frontend import FrontendDemo
from langchain.agents import BaseSingleActionAgent
from langchain.schema import AgentAction
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
import requests
import pytest


class FakeLangchainAgent(BaseSingleActionAgent):
    """Fake LangChain Base Agent."""

    @property
    def input_keys(self):
        """Require property."""
        return ["input"]

    def plan(self, intermediate_steps, **kwargs):
        """Given input, decided what to do."""
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")

    async def aplan(self, intermediate_steps, **kwargs):
        """Given input, decided what to do."""
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")


class FakeAgent:
    """Fake Agent Class."""

    def __init__(self, agent):
        self.agent = agent

    def to_executor(self):
        """Assemble Agent in LCEL and return AgentExecutor instance.

        Returns
        -------
        langchain.agents.AgentExecutor
        """
        agent_executor = AgentExecutor(agent=self.agent, tools=[], memory=ConversationBufferMemory())

        return agent_executor


class TestFrontEnd:
    """Tests for the chatbot/frontend module."""

    @pytest.mark.timeout(5)
    def test_launch(self):
        """Test that frontend launches and returns success status code, using a fake agent."""
        agent = FakeAgent(FakeLangchainAgent())

        demo = FrontendDemo(agent)
        _, local_url, _ = demo.launch(
            share=False, auth=("username", "password"), server_name="0.0.0.0", prevent_thread_lock=True
        )
        response = requests.get(local_url)
        assert response.status_code == 200
        demo.gui.close()
