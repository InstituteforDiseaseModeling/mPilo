import os
import sys
import time

os.environ.setdefault("ENV", "local")
from src.config import OPENAI_MODEL  # noqa: E402  (loads .env and all keys)

from src.retrieval.contextual_retrieval import contextual_retrieval_tool
from src.chatbot.tools import local_tool, human_tool, unknown_tool
from src.chatbot.adaptive_agent import AdaptiveFunctionsAgent
from src.chatbot.prompt import (
    COACH_MPILO_PROMPT_SYSTEM,
    COACH_MPILO_PROMPT_HUMAN_FUNCTION,
    CLASSIFIER_PROMPT,
    SEGMENT_BEHAVIORS,
)
from langchain_openai import ChatOpenAI
import warnings
from langchain._api import LangChainDeprecationWarning
from langchain_community.embeddings import HuggingFaceEmbeddings

warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
VECTORSTORE_PATH = "vectorstore"
RETRIEVER_MODEL_PATH = "gpt4-finetuned-retriever"


print("Tools setup")
who_tool = contextual_retrieval_tool(
    name="search_who_document",
    description="Use this knowledge only to answer questions related to HIV prevention, testing, treatment, service delivery and monitoring",
    documents=None,
    embedding_model=HuggingFaceEmbeddings(model_name=RETRIEVER_MODEL_PATH),
    vectorstore_name=VECTORSTORE_PATH,
)

unknown_place_tool = unknown_tool()
local_place_tool = local_tool()
human_input_tool = human_tool()
tools_list = [who_tool, unknown_place_tool, local_place_tool, human_input_tool]
print("Tools setup completed")

print("Creating agent")
agent = AdaptiveFunctionsAgent(
    tools=tools_list,
    prompt_template_system=COACH_MPILO_PROMPT_SYSTEM,
    prompt_template_human=COACH_MPILO_PROMPT_HUMAN_FUNCTION,
    llm=ChatOpenAI(temperature=0, model_name=OPENAI_MODEL, streaming=True),
    classifier_prompt_template=CLASSIFIER_PROMPT,
    segment_behaviors=SEGMENT_BEHAVIORS,
    translate=True,
)
print("Agent created")

agent_executor = agent.to_executor()
agent_executor.memory.clear()

while True:
    user_input = input("Input Query(input exit to stop): ")
    start = time.time()
    if user_input == "exit":
        print("Exiting")
        sys.exit()
    if user_input == "":
        continue
    result = agent_executor.invoke({"input": user_input})["output"]
    end = time.time()
    total_time = end - start
    print(f"Time taken: {total_time} sec")
    print("Coach:" + result + "\n\n")
