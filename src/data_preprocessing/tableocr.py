"""OCR of tables using multimodal LLM."""

from unstructured.documents.elements import Table
from unstructured.partition.ocr import process_data_with_ocr
import os
import tempfile
from PIL import Image as PILImage
import pdf2image
from openai import AsyncOpenAI
import base64
from io import BytesIO
from bs4 import BeautifulSoup
from src.config import get_secrets, REMOTE_CACHE_S3_BUCKET, TABLE_TO_IMAGE_OPENAI_MODEL
from inspect import signature
import json
import asyncio
import boto3
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


secrets = get_secrets()

LLM_SYSTEM_PROMPT = "You are an expert data analyst and HTML coder."
LLM_PROMPT = "Return an equivalent HTML representation of this table:"
LLM_MAX_TOKENS = 2000
LOCAL_CACHE_PATH = "../../html_cache.json"


async def table_element_llm_html(elements, pdf_filepath, use_cache=True, run_async=False):
    """
    Asynchronously calls OpenAI API on all the Table elements in a PDF document to update their HTML representation.

    Parameters
    ----------
    elements : list
        List of elements in the PDF document.
    pdf_filepath : str
        Path to the PDF file.
    use_cache : bool
        Flag to indicate whether to use the cache.
    run_async : bool
        Flag to indicate whether to use run asynchronously.

    Returns
    -------
    list
        List of elements with updated HTML representation.
    """
    # Make an empty dictionary
    cache = {}
    global cache_updated
    cache_updated = False

    remote_cache = REMOTE_CACHE_S3_BUCKET

    if use_cache and remote_cache:
        s3 = boto3.client("s3")
        try:
            response = s3.get_object(Bucket=remote_cache, Key="html_cache.json")
            cache = json.loads(response["Body"].read())
        except Exception:
            print("html_cache.json not found in remote cache")
        try:
            cache = json.load(open(LOCAL_CACHE_PATH))
        except Exception as e:
            print(f"{e}")
            cache = {}

    # Add images to table elements
    elements = tables_to_images(elements, pdf_filepath)

    # Get all table elements in document.
    table_elements = [element for element in elements if (type(element) == Table)]

    for element in table_elements:
        print(element.table_image)

    async def process_table_element(element):
        """
        Update the HTML representation of a Table element with OpenAI.

        Parameters
        ----------
        element : Table
            The Table element to process.
        """
        filename = element.metadata.filename

        page_number = str(element.metadata.page_number)

        # the element ID is a SHA-256 hash of the element text. There is a small chance this could be duplicated in a document, so we also index by page.
        element_id = element.id

        cached_html = cache.get(filename, {}).get(page_number, {}).get(element_id)

        # If we are using the cache and there is a retrieved value for this table
        if use_cache and cached_html is not None:
            element.metadata.text_as_html = cached_html
        else:
            openai_html = None
            try:
                openai_html = await openai_image_to_html(element.table_image)
                element.metadata.text_as_html = openai_html
            except Exception as e:
                print(f"OpenAI HTML conversion failed on element {element.id}. {e}")

            if use_cache:
                # Then update the cache, creating any nested layers that don't yet exist.
                if openai_html:
                    cache.setdefault(filename, {}).setdefault(page_number, {})[element_id] = openai_html
                global cache_updated
                cache_updated = True

    # Create a list of tasks for each table element
    if run_async:
        tasks = [asyncio.create_task(process_table_element(element)) for element in table_elements]
        # Run the tasks asynchronously
        await asyncio.gather(*tasks)
    else:
        # Run the tasks synchronously
        for element in tqdm(table_elements):
            await process_table_element(element)

    if use_cache and cache_updated:
        # Serializing json
        json_object = json.dumps(cache)

        # Writing to sample.json
        with open(LOCAL_CACHE_PATH, "w") as outfile:
            outfile.write(json_object)

    return elements


def tables_to_images(elements, doc_filepath, dpi=200):
    """
    Add a cropped image to each Table element in a list of elements.

    Parameters
    ----------
    elements : list
        List of elements, including tables.
    doc_filename : str
        Path to the PDF file.
    dpi : int
        Dots per inch for image conversion.

    Returns
    -------
    list
        List of elements with added table images.
    """
    # This must match the dpi that unstructured used to define their coordinates.
    assumed_dpi = signature(process_data_with_ocr).parameters["pdf_image_dpi"].default

    # Create a temporary directory since images are outputted as files.
    with tempfile.TemporaryDirectory() as temp_dir:

        # Convert each page in the PDF into an image, saving them in the temporary directory.
        page_image_paths = pdf2image.convert_from_path(
            doc_filepath,
            dpi=dpi,
            output_folder=temp_dir,
            paths_only=True,
        )
        # For page i+1 in pages
        for i, page_image_path in enumerate(page_image_paths):

            # Get table elements on page
            table_elements = [
                element for element in elements if (type(element) == Table and element.metadata.page_number == i + 1)
            ]

            with PILImage.open(page_image_path) as image:
                for element in table_elements:
                    table_image = crop_image(element, image, dpi, assumed_dpi)
                    # Update the original table object with an image.
                    element.table_image = table_image

    return elements


def crop_image(element, image, dpi, assumed_dpi):
    """
    Crop a page image down to a table image, using its coordinates and scaling by the required dpi.

    Parameters
    ----------
    element : Table
        The Table element.
    image : PILImage.Image
        The page image.
    dpi : int
        Dots per inch for the required image.
    assumed_dpi : int
        Dots per inch assumed by the unstructured module.

    Returns
    -------
    PILImage.Image
        The cropped table image.
    """
    coordinates = element.to_dict()["metadata"]["coordinates"]
    left = coordinates["points"][0][0] / assumed_dpi * dpi
    top = coordinates["points"][0][1] / assumed_dpi * dpi
    right = coordinates["points"][2][0] / assumed_dpi * dpi
    bottom = coordinates["points"][2][1] / assumed_dpi * dpi
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image


# Exponential backoff when API rate limits reached
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def openai_image_to_html(image):
    """
    Use GPT4V to convert an image to HTML.

    Parameters
    ----------
    image : PILImage.Image
        The input image.

    Returns
    -------
    str
        HTML representation of the image content.
    """
    client = AsyncOpenAI(max_retries=5)

    # create an in-memory file object
    output = BytesIO()

    # save the image to the in-memory file object
    image.save(output, format="JPEG")

    # get the bytes of the image from the in-memory file object
    image_bytes = output.getvalue()

    # encode the bytes of the image as base64
    base64_str = base64.b64encode(image_bytes).decode("utf-8")

    print("Requesting HTML from OpenAI...")

    # assemble and send OpenAI API request
    response = await client.chat.completions.create(
        model=TABLE_TO_IMAGE_OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": LLM_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_str}",
                        },
                    },
                ],
            },
        ],
        max_tokens=LLM_MAX_TOKENS,
    )

    text_returned = response.choices[0].message.content

    # Parse the HTML string using BeautifulSoup
    soup = BeautifulSoup(text_returned, "html.parser")

    # Find the table element
    table = soup.find("table")

    # Convert BS object to string
    html_string = str(table)

    print("HTML returned from OpenAI")

    return html_string


def summarise_table(table):
    """
    Return Table summary obtained from an LLM.

    Parameters
    ----------
    table : str
        The Table element's HTML or raw text

    Returns
    -------
    str
        Summary of Table content
    """
    openai.api_key = secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]

    prompt_text = """You are an assistant tasked with summarizing tables. \ 
                    Give a concise summary of the table. Table: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatOpenAI(temperature=0, model="gpt-4")
    summarize_chain = prompt | model | StrOutputParser()
    table_summaries_str = summarize_chain.invoke({"element": table})
    return table_summaries_str
