from PIL import Image as PILImage
from unstructured.documents.elements import Table, CoordinatesMetadata
from unstructured.documents.coordinates import PixelSpace
import pytest
from src.data_preprocessing.tableocr import crop_image, tables_to_images, table_element_llm_html, openai_image_to_html
from copy import deepcopy


class TestTableOCR:
    """Tests for Table OCR functions."""

    @pytest.mark.parametrize("use_cache", [True, False])
    @pytest.mark.asyncio
    async def test_table_element_llm_html(self, test_document_with_elements, mock_openai_response, use_cache):
        """Test that table_element_llm_html function updates the html of all Table elements."""
        # Unpack test set
        doc_filepath, elements = test_document_with_elements

        # Make a deepcopy of elements that won't be modified
        elements_copy = deepcopy(elements)

        elements = await table_element_llm_html(elements=elements, pdf_filepath=doc_filepath, use_cache=use_cache)

        for i, element in enumerate(elements):
            if type(element) == Table:
                # Check that the element id is the same
                assert element.id == elements_copy[i].id
                # Check that the html text has been updated
                assert element.metadata.text_as_html != elements_copy[i].metadata.text_as_html

    @pytest.mark.asyncio
    async def test_openai_image_to_html(image, mock_openai_response):
        """Test that function extracts table HTML from a string returned by the OpenAI API."""
        # Create mock image
        test_image = PILImage.new("RGB", (100, 100), color="white")

        html_string = await openai_image_to_html(test_image)

        # Check string is not empty
        assert len(html_string) > 0
        # Check string contains the table tag
        assert "<table>" in html_string

    def test_tables_to_images(self, test_document_with_elements):
        """Test that the tables_to_images function adds images to Table elements."""
        # Unpack test set
        doc_filepath, elements = test_document_with_elements

        # Make a deepcopy of elements that won't be modified
        elements_copy = deepcopy(elements)

        elements = tables_to_images(elements=elements, doc_filepath=doc_filepath)

        # Check that the original Elements list didn't have images.
        for element in elements_copy:
            if type(element) == Table:
                assert "table_image" not in element.__dict__.keys()
        # Check that all tables now have images.
        for element in elements:
            if type(element) == Table:
                assert type(element.table_image) == PILImage.Image

    def test_crop_image(self):
        """Test the crop image function on mocked image and Table element."""
        # Create mock image
        test_image = PILImage.new("RGB", (100, 100), color="white")

        # Create mock Element
        table = Table(text="table")
        table.metadata.coordinates = CoordinatesMetadata(
            system=PixelSpace(width=100, height=100), points=((10, 10), (90, 10), (90, 90), (10, 90))
        )

        cropped_image = crop_image(table, test_image, dpi=100, assumed_dpi=200)

        # Check image has a white pixel and has some non-zero size
        assert cropped_image.getpixel((0, 0)) == (255, 255, 255)
        assert cropped_image.size[0] > 0
        assert cropped_image.size[1] > 0
