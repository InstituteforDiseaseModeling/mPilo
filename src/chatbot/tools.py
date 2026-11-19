from typing import Callable, Optional, Union, Tuple, Dict

from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool


def local_tool():
    """Tool that searches for places using GoogleSerpAPI."""
    local_search = GoogleSerperAPIWrapper(gl="")
    local_tool_ = Tool(
        name="Local_Search",
        func=local_search.run,
        description="Useful for when you need to search for places. Input should be a search query having a city or place name.",
    )
    return local_tool_


def human_tool():
    """Tool that ask clarifying questions."""
    human_tool_ = Tool(
        name="human_tool", func=HumanInputRunCustom(), description="use this to answer any ambiguous question"
    )
    return human_tool_


def unknown_tool():
    """Tool that ask users location to search for places."""
    unknown_tool_ = Tool(
        name="unknown_place_tool",
        func=HumanInputRunCustom(),
        description="use this when users do not pass any location information about them and ask you to find nearby places",
    )
    return unknown_tool_


def search_tool():
    """Tool that search for any current information from duckduckgo."""
    search = DuckDuckGoSearchRun()
    search_tool_ = Tool(
        name="Current_Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world",
    )
    return search_tool_


def _print_func(text: str) -> None:
    print("\n")
    print(text)


def _return_text(text: str) -> None:
    return text


class HumanInputRunCustom(BaseTool):
    """Tool that asks user for clarifying questions while triggered ambiguity."""

    name: str = "human"
    description: str = (
        "You can ask a human for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be a question for the human."
    )
    prompt_func: Callable[[str], None] = Field(default_factory=lambda: _return_text)

    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        return (), {}

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the Human input tool."""
        query = "You are not sure about the response. Be friendly and ask human to give you some context."
        self.prompt_func(query)

        return query
