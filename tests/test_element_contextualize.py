from src.data_preprocessing.element_contextualize import add_element_context, add_section_metadata


class TestElementContextualize:
    """Tests for the element_contextualize module."""

    def test_add_element_context(self, pdf_elements_to_add_context):
        """Test that the element contextualization adds the correct parameters."""
        elements_plus_context = add_element_context(pdf_elements_to_add_context, stop_title_search_after=4)

        # Test if return > 0
        assert len(elements_plus_context) > 0

        # Test if the elements are given the element_class_plus key
        assert all(["element_class_plus" in element["metadata"] for element in elements_plus_context])

        # Test page num functionality
        assert elements_plus_context[14]["metadata"]["element_class_plus"] == "PageNumber"

        # Test page header functionality
        assert elements_plus_context[15]["metadata"]["element_class_plus"] == "PageHeader"

        # TODO
        # Test page footer functionality

        # TODO
        # Test find main title functionality

    def test_add_section_metadata(self, elements_plus_context):
        """Test that the function correctly assigns elements to the bounding sections."""
        elements_with_sections = add_section_metadata(elements_plus_context)

        # Test if return > 0
        assert len(elements_with_sections) > 0

        # Test if elements are given the header metadata
        must_have = ["section_header1", "section_header2", "section_header3"]
        assert all(
            [all(x in list(element["metadata"].keys()) for x in must_have) for element in elements_with_sections]
        )

        # Test if the headers were populated
        assert len(elements_with_sections[21]["metadata"]["section_header1"]) > 0
