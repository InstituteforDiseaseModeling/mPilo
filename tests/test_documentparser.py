from src.data_preprocessing.documentparser import DocumentParser
from unstructured.documents.elements import Table, NarrativeText, Header, Image, ListItem
from langchain.schema import Document
import pytest
from copy import deepcopy


class TestDocumentParser:
    """Tests for the DocumentParser class."""

    @pytest.mark.parametrize(("document_filename"), [("global_health_p1-2.pdf"), ("who_pages_2_to_10.pdf")])
    def test_from_pdf(self, test_documents, expected_doc, document_filename):
        """Test that the from_pdf returns a list of Langchain documents with expected metadata."""
        dp = DocumentParser(document_character_limit=1000)

        # Retrieve document path in TempDir
        doc_path = test_documents[document_filename]

        pdf_docs = dp.from_pdf(doc_path)

        # Test if length of return is greater than 0
        assert len(pdf_docs) > 0

        # Test if all returned elements are Langchain Docs
        assert all([type(doc) == type(expected_doc) for doc in pdf_docs])

        # Test if metadata keys match expected
        assert all([set(doc.metadata) == set(expected_doc.metadata) for doc in pdf_docs])

    @pytest.mark.parametrize(
        ("document_filename", "should_contain", "should_not_contain"),
        [
            ("who_page_157.pdf", set([Table, NarrativeText, Header]), set([Image])),
            ("global_health_p1-2.pdf", set([NarrativeText, Header, ListItem]), set()),
        ],
    )
    def test_partition_pdf(self, test_documents, document_filename, should_contain, should_not_contain):
        """
        Test that the partition pdf method returns a list of Elements with some expected types.
        """
        dp = DocumentParser(document_character_limit=1000)

        # Retrieve document path in TempDir
        doc_path = test_documents[document_filename]

        # Run method
        elements = dp._partition_pdf(doc_path)

        # Check some elements are returned
        assert len(elements) > 0

        # Get types of each element
        types = set([type(elem) for elem in elements])

        # Check all expected elements are in the elements list
        assert should_contain.issubset(types)

        # Check none of the unexpected elements are in the list
        assert len(should_not_contain.intersection(types)) == 0

    def test_gpt4v_tables(self, test_document_with_elements, mock_openai_response):
        """Test that _gpt4v_tables method updates HTML for Table elements."""
        dp = DocumentParser(document_character_limit=1000)

        # Unpack test set
        doc_filepath, elements = test_document_with_elements

        # Make a deepcopy of elements that won't be modified
        elements_copy = deepcopy(elements)

        elements = dp._gpt4v_tables(elements, doc_filepath)

        for i, element in enumerate(elements):
            if type(element) == Table:
                # Check that the element id is the same
                assert element.id == elements_copy[i].id
                # Check that the html text has been updated
                assert element.metadata.text_as_html != elements_copy[i].metadata.text_as_html

    def test_chunk_table(self, table_html_with_title, table_html_without_title):
        """
        Test that the chunk table method returns a list of Documents with expected Type.
        """
        dp = DocumentParser(document_character_limit=1000)

        # Check for both types of HTML tables

        result_with_title = dp._chunk_table(table_html_with_title, page_number=1)
        result_without_title = dp._chunk_table(table_html_without_title, page_number=1)

        # Assertions for the case with a title
        assert isinstance(result_with_title, list)
        # Check some Documents are returned
        assert len(result_with_title) > 0
        assert isinstance(result_with_title[0], Document)
        assert "title" in result_with_title[0].metadata
        assert "llm_content" in result_with_title[0].metadata
        assert "start_page" in result_with_title[0].metadata
        assert "end_page" in result_with_title[0].metadata

        # Assertions for the case without a title
        assert isinstance(result_without_title, list)
        assert len(result_without_title) > 0
        assert isinstance(result_without_title[0], Document)
        assert "title" not in result_without_title[0].metadata
        assert "llm_content" in result_without_title[0].metadata
        assert "start_page" in result_without_title[0].metadata
        assert "end_page" in result_without_title[0].metadata

    def test_merge_text_section(self, elements_with_sections):
        """Test that the function correctly merge elements based on bounding sections."""
        dp = DocumentParser(document_character_limit=1000)
        text_in_sections = dp._merge_text_section(elements_with_sections)

        # Test if function return is > 0
        assert len(text_in_sections) > 0

        # Test if text strings in Document has length > 0
        assert all([len(doc.page_content) > 0 for doc in text_in_sections])

        # Test if element returned has the correct metadata attributes
        must_have = [
            "element_class",
            "start_page",
            "end_page",
            "section_header1",
            "section_header2",
            "section_header3",
        ]
        assert all([all(x in doc.metadata for x in must_have) for doc in text_in_sections])

    @pytest.mark.parametrize(
        ("character_limits"),
        [100, 200, 500, 1000],
    )
    def test_chunk_text(self, text_in_sections, character_limits):
        """Test that the function correctly splits documents to have character count less or equal to given limit."""
        dp = DocumentParser(document_character_limit=character_limits)
        text_docs = dp._chunk_text(text_in_sections, document_character_limit=character_limits)

        # Test if function return is > 0
        assert len(text_docs) > 0

        # Test if page_content length is less than given token limit
        assert all([len(doc.page_content) <= character_limits for doc in text_docs])

        # Test if element returned has the correct metadata attributes
        must_have = [
            "element_class",
            "start_page",
            "end_page",
            "section_header1",
            "section_header2",
            "section_header3",
            "llm_content",
        ]
        assert all([all(x in doc.metadata for x in must_have) for doc in text_docs])
