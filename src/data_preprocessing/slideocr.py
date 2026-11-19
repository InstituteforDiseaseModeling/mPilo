from openai import OpenAI
import os
from io import BytesIO
import base64
import pdf2image
from src.config import get_secrets, TABLE_TO_IMAGE_OPENAI_MODEL
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

get_secrets()

LLM_SYSTEM_PROMPT = "You are an expert data analyst and transcriber."
LLM_PROMPT = "Transcribe the text from this slide. If it contains diagrams, communicate the information from the diagrams in text form. Only return the slide in text form."
DPI = 200
LLM_MAX_TOKENS = 4000


def openai_page_to_text(doc_filepath, page_number, filetype="pdf"):
    """
    Use OpenAI's API with a multimodal model to generate a text representation of a page from a document.

    Parameters
    ----------
    doc_filepath : str
        Path to the document file.
    page_number : int
        Number of requested page in document.
    filetype : str
        Filetype of document - currently only supports "pdf".

    Returns
    -------
    str
        Text representation of the page.
    """
    if filetype == "pdf":
        page_image = pdf_page_to_image(doc_filepath=doc_filepath, page_number=page_number)

    page_text = openai_page_image_to_text(page_image=page_image)

    return page_text


def pdf_page_to_image(doc_filepath, page_number):
    """
    Return an image of a full PDF page.

    Parameters
    ----------
    doc_filepath : str
        Path to the PDF file.
    page_number : int
        Number of requested page in PDF.

    Returns
    -------
    PILImage.Image
        The page image.
    """
    # Convert each page in the PDF into an image, saving them in the temporary directory.
    page_images = pdf2image.convert_from_path(doc_filepath, dpi=DPI)

    return page_images[page_number - 1]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_page_image_to_text(page_image):
    """
    Use GPT4V to return a text representation of a page including diagrams.

    Parameters
    ----------
    page_image : PILImage.Image
        The input image of the slide.

    Returns
    -------
    str
        Text representation of the page content.
    """
    client = OpenAI(max_retries=5)

    # create an in-memory file object
    output = BytesIO()

    # save the image to the in-memory file object
    page_image.save(output, format="JPEG")

    # get the bytes of the image from the in-memory file object
    image_bytes = output.getvalue()

    # encode the bytes of the image as base64
    base64_str = base64.b64encode(image_bytes).decode("utf-8")

    print("Requesting text from OpenAI...")

    # assemble and send OpenAI API request
    response = client.chat.completions.create(
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

    return text_returned


def clean_text(elements, include_pages):
    """
    Clean Unstructured text elements to remove page numbers and empty elements, returning a single string.

    Parameters
    ----------
    elements : list
        List of Element objects to clean.

    Returns
    -------
    str
        Single string containing cleaned text from all elements.
    """
    # Filter out pages not in the include_pages list
    filtered_elements = [el for el in elements if el.metadata.page_number in include_pages]

    # Clean text from elements to remove page numbers, empty strings and misc text.
    text = " ".join(
        [
            " ".join(str(el).split())
            for el in filtered_elements
            if not (str(el).isnumeric() or str(el) == "<number>" or len(str(el)) == 0)
        ]
    )
    return text
