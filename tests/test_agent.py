from src.chatbot.adaptive_agent import AdaptiveReActAgent, AdaptiveFunctionsAgent
from src.chatbot.agent import ReActAgent, FunctionsAgent
from src.chatbot.prompt import (
    COACH_MPILO_PROMPT_SYSTEM,
    COACH_MPILO_PROMPT_HUMAN_FUNCTION,
    COACH_MPILO_PROMPT_HUMAN,
    CLASSIFIER_PROMPT,
    SEGMENT_BEHAVIORS,
)
from src.retrieval.contextual_retrieval import contextual_retrieval_tool
from src.chatbot.tools import local_tool, human_tool, unknown_tool
from langchain.llms.fake import FakeListLLM
from langchain_community.chat_models.fake import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


class TestAgent:
    """Tests for the chatbot/agent module."""

    def test_react_agent(self, expected_doc):
        """Tests the ReActAgent using a mocked LLM."""
        # Replace the llm instantiation with FakeListLLM
        fake_llm = FakeListLLM(
            responses=[
                """Do I need to use a tool? No
        Final Answer: Hello, I'm doing great thanks. How are you?"""
            ]
        )

        # Create mocked retrieval tool using single mocked Document.
        who_tool = contextual_retrieval_tool(
            documents=[expected_doc],
        )
        unknown_place_tool = unknown_tool()
        local_place_tool = local_tool()
        human_input_tool = human_tool()
        tools = [who_tool, unknown_place_tool, local_place_tool, human_input_tool]

        # Create an instance of ReActAgent
        react_agent = ReActAgent(
            tools,
            llm=fake_llm,
            prompt_template_system=COACH_MPILO_PROMPT_SYSTEM,
            prompt_template_human=COACH_MPILO_PROMPT_HUMAN,
        )

        # Get an AgentExecutor instance from the ReActAgent instance
        agent_executor = react_agent.to_executor()

        # Use the invoke() method to interact with the agent and test its functionality
        output = agent_executor.invoke({"input": "Hi, how are you?"})
        assert output["output"] == "Hello, I'm doing great thanks. How are you?"

    def test_functions_agent(self, expected_doc):
        """Tests the FunctionsAgent using a mocked LLM."""
        # Replace the llm instantiation with FakeListLLM
        fake_llm = FakeMessagesListChatModel(
            responses=[AIMessage(content="""Hello, I'm doing great thanks. How are you?""")]
        )

        # Create mocked retrieval tool using single mocked Document.
        who_tool = contextual_retrieval_tool(
            name="search_who_document",
            description="Search 'Consolidated guidelines on HIV prevention, testing, treatment, service delivery and monitoring' document.",
            documents=[expected_doc],
        )
        unknown_place_tool = unknown_tool()
        local_place_tool = local_tool()
        human_input_tool = human_tool()
        tools = [who_tool, unknown_place_tool, local_place_tool, human_input_tool]

        # Create an instance of FunctionsAgent
        react_agent = FunctionsAgent(
            tools,
            llm=fake_llm,
            prompt_template_system=COACH_MPILO_PROMPT_SYSTEM,
            prompt_template_human=COACH_MPILO_PROMPT_HUMAN_FUNCTION,
        )

        # Get an AgentExecutor instance from the ReActAgent instance
        agent_executor = react_agent.to_executor()

        # Use the invoke() method to interact with the agent and test its functionality
        output = agent_executor.invoke({"input": "Hi, how are you?"})
        assert output["output"] == "Hello, I'm doing great thanks. How are you?"

    def test_adaptive_react_agent(self, expected_doc):
        """Tests the AdaptiveReActAgent using a mocked LLM."""
        # Replace the llm instantiation with FakeListLLM

        fake_llm = FakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content="""Do I need to use a tool? No
        Final Answer: Hello, I'm doing great thanks. How are you?"""
                )
            ]
        )

        fake_classifier_llm = FakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content='{\n    "Teal": 1,\n    "Rose": 0,\n    "Green": 0,\n    "Blue": 0,\n    "Gray": 0,\n    "Opaque": 0\n}'
                )
            ]
        )

        # Create mocked retrieval tool using single mocked Document.
        who_tool = contextual_retrieval_tool(
            name="search_who_document",
            description="Search 'Consolidated guidelines on HIV prevention, testing, treatment, service delivery and monitoring' document.",
            documents=[expected_doc],
        )
        unknown_place_tool = unknown_tool()
        local_place_tool = local_tool()
        human_input_tool = human_tool()
        tools = [who_tool, unknown_place_tool, local_place_tool, human_input_tool]

        # Create an instance of AdaptiveReActAgent
        react_agent = AdaptiveReActAgent(
            tools,
            llm=fake_llm,
            classifier_llm=fake_classifier_llm,
            prompt_template_system=COACH_MPILO_PROMPT_SYSTEM,
            prompt_template_human=COACH_MPILO_PROMPT_HUMAN,
            classifier_prompt_template=CLASSIFIER_PROMPT,
            segment_behaviors=SEGMENT_BEHAVIORS,
        )

        # Get an AgentExecutor instance from the ReActAgent instance
        agent_executor = react_agent.to_executor()

        # Use the invoke() method to interact with the agent and test its functionality
        output = agent_executor.invoke({"input": "Hi, how are you?"})
        assert output["output"] == "Hello, I'm doing great thanks. How are you?"

    def test_adaptive_functions_agent(self, expected_doc):
        """Tests the AdaptiveFunctionsAgent using a mocked LLM."""
        # Replace the llm instantiation with FakeListLLM

        fake_llm = FakeMessagesListChatModel(
            responses=[AIMessage(content="""Hello, I'm doing great thanks. How are you?""")]
        )

        fake_classifier_llm = FakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content='{\n    "Teal": 1,\n    "Rose": 0,\n    "Green": 0,\n    "Blue": 0,\n    "Gray": 0,\n    "Opaque": 0\n}'
                )
            ]
        )

        # Create mocked retrieval tool using single mocked Document.
        who_tool = contextual_retrieval_tool(
            name="search_who_document",
            description="Search 'Consolidated guidelines on HIV prevention, testing, treatment, service delivery and monitoring' document.",
            documents=[expected_doc],
        )
        unknown_place_tool = unknown_tool()
        local_place_tool = local_tool()
        human_input_tool = human_tool()
        tools = [who_tool, unknown_place_tool, local_place_tool, human_input_tool]

        # Create an instance of AdaptiveFunctionsAgent
        react_agent = AdaptiveFunctionsAgent(
            tools,
            llm=fake_llm,
            classifier_llm=fake_classifier_llm,
            prompt_template_system=COACH_MPILO_PROMPT_SYSTEM,
            prompt_template_human=COACH_MPILO_PROMPT_HUMAN,
            classifier_prompt_template=CLASSIFIER_PROMPT,
            segment_behaviors=SEGMENT_BEHAVIORS,
        )

        # Get an AgentExecutor instance from the ReActAgent instance
        agent_executor = react_agent.to_executor()

        # Use the invoke() method to interact with the agent and test its functionality
        output = agent_executor.invoke({"input": "Hi, how are you?"})
        assert output["output"] == "Hello, I'm doing great thanks. How are you?"
