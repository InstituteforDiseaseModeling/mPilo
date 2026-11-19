from src.config import (
    get_secrets,
    DOCS_S3_BUCKET,
    IS_LOCAL,
    IS_PRODUCTION,
    OPENAI_MODEL,
    REDIS_PORT,
    REDIS_SERVER_ADDRESS,
)
from src.chatbot.adaptive_agent import AdaptiveFunctionsAgent
from src.chatbot.prompt import (
    COACH_MPILO_PROMPT_SYSTEM,
    COACH_MPILO_PROMPT_HUMAN_FUNCTION,
    CLASSIFIER_PROMPT,
    SEGMENT_BEHAVIORS,
)
from src.chatbot.segmentation import freq_prob
from src.data_preprocessing.documentparser import DocumentParser
from src.retrieval.contextual_retrieval import contextual_retrieval_tool
from src.server.redis_utils import (
    ingest_to_db,
    retrieve_from_db,
    SHORT_TERM_MEMORY_KEY,
    LONG_TERM_MEMORY_KEY,
    USER_DETAILS_KEY,
    USER_SEGMENT_PROBS_KEY,
)
from src.chatbot.tools import local_tool, human_tool, unknown_tool

from langchain_openai import ChatOpenAI
from langchain.schema import messages_from_dict, messages_to_dict
from fastapi import FastAPI, Form, Response, Request, HTTPException
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
from langchain_community.embeddings import HuggingFaceEmbeddings
import boto3
import pickle
import redis
import pandas as pd
import os


secrets = get_secrets()
account_sid = secrets.get("TWILIO_ACCOUNT_SID")
auth_token = secrets.get("TWILIO_AUTH_TOKEN")

script_dir = os.path.dirname(os.path.abspath(__file__))
SOURCE_ELEMENTS = "9789240031593-eng.elements_gpt4v"
WHATSAPP_MODEL = OPENAI_MODEL
VECTORSTORE_PATH = os.path.join(script_dir, "../../vectorstore")
RETRIEVER_MODEL_PATH = os.path.join(script_dir, "../../gpt4-finetuned-retriever")


def load_documents():
    """Load elements list from S3."""
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=DOCS_S3_BUCKET, Key=SOURCE_ELEMENTS)
    elements = pickle.loads(response["Body"].read())
    doc_parser = DocumentParser(document_character_limit=2000)
    documents = doc_parser.from_elements(elements)
    return documents


print("Loading retrieval docs...")
if not IS_LOCAL:
    documents = load_documents()
    who_tool = contextual_retrieval_tool(documents=documents)
else:
    who_tool = contextual_retrieval_tool(
        name="search_who_document",
        description="Use this knowledge only to answer questions related to HIV prevention, testing, treatment, service delivery and monitoring",
        documents=None,
        embedding_model=HuggingFaceEmbeddings(model_name=RETRIEVER_MODEL_PATH),
        vectorstore_name=VECTORSTORE_PATH,
    )

print("Retrieval docs loaded")

print("Creating tools")
unknown_place_tool = unknown_tool()
local_place_tool = local_tool()
human_input_tool = human_tool()
print("Tools created")

print("Creating agent")
agent = AdaptiveFunctionsAgent(
    tools=[who_tool, unknown_place_tool, local_place_tool, human_input_tool],
    prompt_template_system=COACH_MPILO_PROMPT_SYSTEM,
    prompt_template_human=COACH_MPILO_PROMPT_HUMAN_FUNCTION,
    llm=ChatOpenAI(temperature=0, model_name=WHATSAPP_MODEL, streaming=True),
    classifier_prompt_template=CLASSIFIER_PROMPT,
    segment_behaviors=SEGMENT_BEHAVIORS,
    translate=True,
)
print("Agent created")

app = FastAPI(docs_url=None, redoc_url=None)
rds = redis.Redis(host=REDIS_SERVER_ADDRESS, port=REDIS_PORT, decode_responses=True)


async def validate_request(request):
    """Validate that a request comes from Twilio."""
    validator = RequestValidator(secrets["TWILIO_AUTH_TOKEN"])
    form_ = await request.form()
    if not validator.validate(str(request.url), form_, request.headers.get("X-Twilio-Signature", "")):
        raise HTTPException(status_code=400, detail="Error in Twilio Signature")


@app.post("/hook")
async def chat(request: Request, From: str = Form(...), Body: str = Form(...)):
    """Return an output for a given prompt."""
    # Validate request, if in production
    if IS_PRODUCTION:
        await validate_request(request)

    # For debugging
    # print(rds.keys())

    # Check whether there are redis keys associated with this phone number
    # Check SHORT_TERM_MEMORY_KEY
    if rds.exists(f"{SHORT_TERM_MEMORY_KEY}/{From}"):
        agent.memory.chat_memory.messages = messages_from_dict(
            retrieve_from_db(client=rds, prefix=SHORT_TERM_MEMORY_KEY, user=From)
        )
        print("Retrieved memory")

    # Check USER_SEGMENT_PROBS_KEY
    if rds.exists(f"{USER_SEGMENT_PROBS_KEY}/{From}"):
        agent.segment_probs = pd.DataFrame.from_dict(
            retrieve_from_db(client=rds, prefix=USER_SEGMENT_PROBS_KEY, user=From)
        )
        print("Retrieved segment probs")

    # Check USER_DETAILS_KEY
    if rds.exists(f"{USER_DETAILS_KEY}/{From}"):
        user_details = retrieve_from_db(client=rds, prefix=USER_DETAILS_KEY, user=From)
        print("Retrieved user details")
        agent.inject_user_details(user_name=From, user_details=user_details, silent=False)

    # For debugging
    # print(str(Body))
    if str(Body).lower() == "clear memory":
        response = MessagingResponse()

        # Clear memory object
        print("Clearing memory")
        agent.memory.clear()

        # Clear segment probabilities
        print("Reseting probabilities")
        _ = agent.segment_probs = freq_prob(neutral_weight=0)

        # Clear redis objects
        print("Clearing user entries on redis")
        ingest_to_db(client=rds, prefix=SHORT_TERM_MEMORY_KEY, user=From, _dict="")
        ingest_to_db(client=rds, prefix=LONG_TERM_MEMORY_KEY, user=From, _dict="")
        ingest_to_db(client=rds, prefix=USER_DETAILS_KEY, user=From, _dict="")
        ingest_to_db(client=rds, prefix=USER_SEGMENT_PROBS_KEY, user=From, _dict=agent.segment_probs.to_dict())

        _ = response.message("Memory cleared")
        return Response(content=str(response), media_type="application/xml")

    # print(agent.memory)
    print(f"Hi {From}, you said: {Body}")
    reply = agent.to_executor().invoke({"input": Body})["output"]
    # Attempt to crop reply so total length doesn't exceed 1600 chars
    reply = reply[:1500]
    response = MessagingResponse()
    _ = response.message(reply)

    # Save updated short term memory and segment probabilities concurrently
    ingest_to_db(
        client=rds, prefix="short_term_memory", user=From, _dict=messages_to_dict(agent.memory.chat_memory.messages)
    )
    ingest_to_db(client=rds, prefix="user_segment_probs", user=From, _dict=agent.segment_probs.to_dict())
    return Response(content=str(response), media_type="application/xml")


@app.get("/health")
def health():
    """Health check."""
    print("health endpoint")
