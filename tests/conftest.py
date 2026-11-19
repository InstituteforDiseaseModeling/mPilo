from src.config import DOCS_S3_BUCKET
"""Makes fixtures available to all tests in the directory."""

import boto3
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from datetime import datetime
from unittest.mock import AsyncMock
from langchain.schema import Document
import pytest
import pickle


@pytest.fixture
def expected_doc():
    """Generate expected Langchain output format to test PDF extraction."""
    mock_lang_doc = Document(
        page_content="Page content.",
        metadata={
            "element_class": "Element class",
            "start_page": 0,
            "end_page": 0,
            "section_header1": "Section header",
            "section_header2": "Section header",
            "section_header3": "Section header",
            "llm_content": "LLM content.",
        },
    )

    return mock_lang_doc


@pytest.fixture
def test_document_with_elements(tmp_path):
    """Gather unstructured data examples for tests and store in tempdir."""
    # Expected to be present in the s3://$DOCS_S3_BUCKET/test_docs/ folder.
    BUCKET_NAME = DOCS_S3_BUCKET
    FOLDER = "test_docs"
    DOCUMENT_FILENAME = "who_page_157.pdf"
    ELEMENTS_FILENAME = "who_page_157.elements"
    s3 = boto3.resource("s3")

    # Download the document file to a temporary directory
    local_path = tmp_path / DOCUMENT_FILENAME
    s3_key = f"{FOLDER}/{DOCUMENT_FILENAME}"
    s3.meta.client.download_file(BUCKET_NAME, s3_key, local_path)
    doc_path = str(local_path)

    # Download the elements list
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=BUCKET_NAME, Key=f"{FOLDER}/{ELEMENTS_FILENAME}")
    elements_bytes = response["Body"].read()
    elements = pickle.loads(elements_bytes)

    return doc_path, elements


@pytest.fixture
def mock_openai_response(mocker):
    """Mock the response from the OpenAI API to save cost."""
    mock_response = ChatCompletion(
        id="foo",
        model="gpt-4",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="""Here is some HTML:<table>
                                          <tr>
                                            <th>Column 1</th>
                                            <th>Column 2</th>
                                            <th>Column 3</th>
                                          </tr>
                                          <tr>
                                            <td>Row 1, Column 1</td>
                                            <td>Row 1, Column 2</td>
                                            <td>Row 1, Column 3</td>
                                          </tr>
                                          <tr>
                                            <td>Row 2, Column 1</td>
                                            <td>Row 2, Column 2</td>
                                            <td>Row 2, Column 3</td>
                                          </tr>
                                        </table>
                                        """,
                    role="assistant",
                ),
                logprobs=None,
            )
        ],
        created=int(datetime.now().timestamp()),
    )
    # create a mock client object
    mock_client = AsyncMock()

    # set the return value of the mock client object to the mock response object
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    mocker.patch("src.data_preprocessing.tableocr.AsyncOpenAI", return_value=mock_client)
    return mocker


@pytest.fixture
def mock_openai_agent_response(mocker):
    """Mock the response from the OpenAI API to save cost."""
    mock_response = ChatCompletion(
        id="foo",
        model="gpt-4",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="""Here is some HTML:<table>
                                          <tr>
                                            <th>Column 1</th>
                                            <th>Column 2</th>
                                            <th>Column 3</th>
                                          </tr>
                                          <tr>
                                            <td>Row 1, Column 1</td>
                                            <td>Row 1, Column 2</td>
                                            <td>Row 1, Column 3</td>
                                          </tr>
                                          <tr>
                                            <td>Row 2, Column 1</td>
                                            <td>Row 2, Column 2</td>
                                            <td>Row 2, Column 3</td>
                                          </tr>
                                        </table>
                                        """,
                    role="assistant",
                ),
            )
        ],
        created=int(datetime.now().timestamp()),
    )
    # create a mock client object
    mock_client = AsyncMock()

    # set the return value of the mock client object to the mock response object
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    mocker.patch("src.data_preprocessing.tableocr.AsyncOpenAI", return_value=mock_client)
    return mocker


@pytest.fixture
def test_documents(tmp_path):
    """Gather unstructured data examples for tests and store in tempdir."""
    # Expected to be present in the s3://$DOCS_S3_BUCKET/test_docs/ folder.
    BUCKET_NAME = DOCS_S3_BUCKET
    FOLDER = "test_docs"
    DOCUMENT_FILENAMES = ["global_health_p1-2.pdf", "hiv_therapy_p1-2.pdf", "who_page_157.pdf", "who_pages_2_to_10.pdf"]
    s3 = boto3.resource("s3")
    doc_paths = {}
    for doc_file in DOCUMENT_FILENAMES:
        # Download the file to a temporary directory
        local_path = tmp_path / doc_file
        s3_key = f"{FOLDER}/{doc_file}"
        s3.meta.client.download_file(BUCKET_NAME, s3_key, local_path)
        doc_paths[doc_file] = str(local_path)
    return doc_paths


@pytest.fixture
def table_html_with_title():
    """Return Table HTML with Title."""
    return "<caption>Table Title</caption><table><tr><th>Header1</th><th>Header2</th></tr><tr><td>Data1</td><td>Data2</td></tr></table>"


@pytest.fixture
def table_html_without_title():
    """Return Table HTML without Title."""
    return "<table><tr><th>Header1</th><th>Header2</th></tr><tr><td>Data1</td><td>Data2</td></tr></table>"


@pytest.fixture
def pdf_elements_to_add_context():
    """Gather unstructured data examples for tests."""
    # Expected to be present in the s3://$DOCS_S3_BUCKET/test_docs/ folder.
    BUCKET_NAME = DOCS_S3_BUCKET
    FOLDER = "test_docs"
    ELEMENTS_FILENAME = "who_pages_16_to_22.elements"
    s3 = boto3.resource("s3")

    # Download the elements list
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=BUCKET_NAME, Key=f"{FOLDER}/{ELEMENTS_FILENAME}")
    elements_bytes = response["Body"].read()
    elements = pickle.loads(elements_bytes)

    return elements


@pytest.fixture
def elements_plus_context():
    """Gather elements with class context examples for tests."""
    # Expected to be present in the s3://$DOCS_S3_BUCKET/test_docs/ folder.
    BUCKET_NAME = DOCS_S3_BUCKET
    FOLDER = "test_docs"
    ELEMENTS_FILENAME = "who_pages_16_to_22.el_context"
    s3 = boto3.resource("s3")

    # Download the elements list
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=BUCKET_NAME, Key=f"{FOLDER}/{ELEMENTS_FILENAME}")
    elements_bytes = response["Body"].read()
    elements_plus_context = pickle.loads(elements_bytes)

    return elements_plus_context


@pytest.fixture
def elements_with_sections():
    """Gather Documents merged in sections for tests."""
    # Expected to be present in the s3://$DOCS_S3_BUCKET/test_docs/ folder.
    BUCKET_NAME = DOCS_S3_BUCKET
    FOLDER = "test_docs"
    ELEMENTS_FILENAME = "who_pages_16_to_22.el_with_sections"
    s3 = boto3.resource("s3")

    # Download the elements list
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=BUCKET_NAME, Key=f"{FOLDER}/{ELEMENTS_FILENAME}")
    elements_bytes = response["Body"].read()
    elements_with_sections = pickle.loads(elements_bytes)

    return elements_with_sections


@pytest.fixture
def text_in_sections():
    """Gather Documents merged in sections for tests."""
    # Expected to be present in the s3://$DOCS_S3_BUCKET/test_docs/ folder.
    BUCKET_NAME = DOCS_S3_BUCKET
    FOLDER = "test_docs"
    ELEMENTS_FILENAME = "who_pages_16_to_22.text_in_sections"
    s3 = boto3.resource("s3")

    # Download the elements list
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=BUCKET_NAME, Key=f"{FOLDER}/{ELEMENTS_FILENAME}")
    elements_bytes = response["Body"].read()
    text_in_sections = pickle.loads(elements_bytes)

    return text_in_sections


@pytest.fixture
def mr_rose_conversation_buffer():
    """Gather Documents merged in sections for tests."""
    # Expected to be present in the s3://$DOCS_S3_BUCKET/test_docs/ folder.
    BUCKET_NAME = DOCS_S3_BUCKET
    FOLDER = "mock_chat_history"
    ELEMENTS_FILENAME = "all_segments.pickle"
    s3 = boto3.resource("s3")

    # Download the elements list
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=BUCKET_NAME, Key=f"{FOLDER}/{ELEMENTS_FILENAME}")
    elements_bytes = response["Body"].read()
    conversations = pickle.loads(elements_bytes)
    mr_rose = conversations["rose"]

    return mr_rose
