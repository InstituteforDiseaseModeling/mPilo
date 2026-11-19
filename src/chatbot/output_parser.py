from langchain.agents import AgentOutputParser
from langchain_core.agents import AgentActionMessageLog
from langchain.schema import AgentAction, AgentFinish
from langchain_core.messages import AIMessage
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from typing import Union
import re
import json
from json import JSONDecodeError


class CustomReActOutputParser(AgentOutputParser):
    """Customize output for the llms."""

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """Parse tools, action and action-inputs from the logs."""
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            agent_response = llm_output.split("Final Answer:")[-1].strip()

            response_end = agent_response.find("\n```")
            if response_end < 0:
                response_end = agent_response.find("```")

            if response_end >= 0:
                agent_response = agent_response[:response_end]

            return AgentFinish(
                return_values={"output": agent_response},
                log=llm_output,
            )

        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        if not match:
            output = f"Check the LLM output for the response: `{llm_output}`"
            action = ""
            action_input = ""
        else:
            output = llm_output
            action = match.group(1).strip()
            action_input = match.group(2)

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=output)


class FunctionaryOutputParser(OpenAIFunctionsAgentOutputParser):
    """Modify the OpenAI output parser for compatibility with Functionary."""

    @staticmethod
    def _parse_ai_message(message):
        """Parse an AI message."""
        if not isinstance(message, AIMessage):
            raise TypeError(f"Expected an AI message got {type(message)}")

        tool_calls = message.additional_kwargs.get("tool_calls", {})

        if tool_calls:
            function_call = tool_calls[0]["function"]
            function_name = function_call["name"]
            try:
                if len(function_call["arguments"].strip()) == 0:
                    # OpenAI returns an empty string for functions containing no args
                    _tool_input = {}
                else:
                    # otherwise it returns a json object
                    _tool_input = json.loads(function_call["arguments"], strict=False)
            except JSONDecodeError:
                raise OutputParserException(
                    f"Could not parse tool input: {function_call} because " f"the `arguments` is not valid JSON."
                )

            # HACK HACK HACK:
            # The code that encodes tool input into Open AI uses a special variable
            # name called `__arg1` to handle old style tools that do not expose a
            # schema and expect a single string argument as an input.
            # We unpack the argument here if it exists.
            # Open AI does not support passing in a JSON array as an argument.
            if "__arg1" in _tool_input:
                tool_input = _tool_input["__arg1"]
            else:
                tool_input = _tool_input

            content_msg = f"responded: {message.content}\n" if message.content else "\n"
            log = f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n"
            return AgentActionMessageLog(
                tool=function_name,
                tool_input=tool_input,
                log=log,
                message_log=[message],
            )

        return AgentFinish(return_values={"output": message.content}, log=str(message.content))
