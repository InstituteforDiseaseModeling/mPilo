from langchain.agents import Tool
from src.chatbot.tools import HumanInputRunCustom


class TestTools:
    """Tests for the tools."""

    def test_human_input_tool(self):
        """Tests the human input tool for ambiguous question."""
        # initialize human tool
        human_tool = Tool(
            name="human tool", func=HumanInputRunCustom(), description="use this to answer any ambiguous question"
        )

        # user's ambiguous question
        user_input = "Could you please tell me about the treatment of my disease?"

        # get the response
        response = human_tool.run(user_input)

        assert response == "You are not sure about the response. Be friendly and ask human to give you some context."
