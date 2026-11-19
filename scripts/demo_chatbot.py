"""Launches demo chatbot server."""
from src.data_preprocessing.documentparser import DocumentParser
from src.chatbot.frontend import FrontendDemo
from src.chatbot.adaptive_agent import AdaptiveFunctionsAgent
from src.chatbot.prompt import (
    COACH_MPILO_PROMPT_SYSTEM,
    COACH_MPILO_PROMPT_HUMAN_FUNCTION,
    CLASSIFIER_PROMPT,
    SEGMENT_BEHAVIORS,
)
from src.retrieval.contextual_retrieval import contextual_retrieval_tool
from src.chatbot.tools import local_tool, human_tool, unknown_tool
from langchain_openai import ChatOpenAI
from src.config import DOCS_S3_BUCKET, OPENAI_MODEL, demo_auth
import boto3
import pickle
import langchain

langchain.debug = False

SOURCE_ELEMENTS = "9789240031593-eng.elements_gpt4v"
DEMO_MODEL = OPENAI_MODEL


def load_documents():
    """Load elements list from S3."""
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=DOCS_S3_BUCKET, Key=SOURCE_ELEMENTS)
    elements = pickle.loads(response["Body"].read())
    doc_parser = DocumentParser(document_character_limit=2000)
    documents = doc_parser.from_elements(elements)
    return documents


print("Loading retrieval docs...")
documents = load_documents()
print("Retrieval docs loaded")

print("Creating tools")
who_tool = contextual_retrieval_tool(documents=documents)
unknown_place_tool = unknown_tool()
local_place_tool = local_tool()
human_input_tool = human_tool()
print("Tools created")

print("Creating agent")
agent = AdaptiveFunctionsAgent(
    tools=[who_tool, unknown_place_tool, local_place_tool, human_input_tool],
    prompt_template_system=COACH_MPILO_PROMPT_SYSTEM,
    prompt_template_human=COACH_MPILO_PROMPT_HUMAN_FUNCTION,
    llm=ChatOpenAI(temperature=0, model_name=DEMO_MODEL, streaming=True),
    classifier_prompt_template=CLASSIFIER_PROMPT,
    segment_behaviors=SEGMENT_BEHAVIORS,
    translate=True,
)
print("Agent created")

print("Launching frontend")
demo = FrontendDemo(agent)
demo.launch(share=True, auth=demo_auth(), server_name="0.0.0.0")
