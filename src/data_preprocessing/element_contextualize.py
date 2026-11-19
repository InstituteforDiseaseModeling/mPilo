"""Finetuning of Ustructured Classes."""

import re


def _is_page_header(element):
    """Find if element is a page header.

    Parameters
    ----------
    element : Unstructured Element dictionary
        The element to be checked.

    Returns
    -------
    bool
        Returns True if the element is likely to be a page header, False otherwise.

    """
    is_page_header = False

    # If element is already classified as page number, skip
    if element["metadata"]["element_class_plus"] == "PageNumber":
        return is_page_header

    # Detect if element is a page footer
    page_width = element["metadata"]["coordinates"]["layout_width"]
    page_height = element["metadata"]["coordinates"]["layout_height"]

    # The header should occupy the ratio % of the page top
    header_ratio = 0.07

    header_box = {"x": 0, "y": 0, "height": page_height * header_ratio, "width": page_width}

    # Define element coordinates
    coords = element["metadata"]["coordinates"]["points"]
    element_box = {
        "x": max(coords[0][0], 0),  # Unstructured sometimes assign elements to negative coordinates
        "y": max(coords[0][1], 0),  # Unstructured sometimes assign elements to negative coordinates
        "height": coords[1][1] - max(coords[0][1], 0),
        "width": min(coords[2][0] - coords[0][0], page_width),
    }

    # Find if element is contained in the header_box
    # If top-left inner box corner is inside the bounding box
    if header_box["x"] <= element_box["x"] and header_box["y"] <= element_box["y"]:
        # If bottom-right inner box corner is inside the bounding box
        if (
            element_box["x"] + element_box["width"] <= header_box["x"] + header_box["width"]
            and element_box["y"] + element_box["height"] <= header_box["y"] + header_box["height"]
        ):
            is_page_header = True

    return is_page_header


def _is_page_footer(element):
    """Find if element is a page footer.

    Parameters
    ----------
    element : Unstructured Element dictionary
        The element to be checked.

    Returns
    -------
    bool
        Returns True if the element is likely to be a page footer, False otherwise.

    """
    is_page_footer = False

    # If element is already classified as page number, skip
    if element["metadata"]["element_class_plus"] == "PageNumber":
        return is_page_footer

    # Detect if element is a page footer
    page_width = element["metadata"]["coordinates"]["layout_width"]
    page_height = element["metadata"]["coordinates"]["layout_height"]

    # The footer should occupy the ratio % of the page bottom
    footer_ratio = 0.1
    inv_ratio = 1 - footer_ratio

    footer_box = {"x": 0, "y": page_height * inv_ratio, "height": page_height * footer_ratio, "width": page_width}

    # Define element coordinates
    coords = element["metadata"]["coordinates"]["points"]
    element_box = {
        "x": coords[0][0],
        "y": coords[0][1],
        "height": coords[1][1] - coords[0][1],
        "width": coords[2][0] - coords[0][0],
    }

    # Find if element is contained in the footer_box
    # If top-left inner box corner is inside the bounding box
    if footer_box["x"] <= element_box["x"] and footer_box["y"] <= element_box["y"]:
        # If bottom-right inner box corner is inside the bounding box
        if (
            element_box["x"] + element_box["width"] <= footer_box["x"] + footer_box["width"]
            and element_box["y"] + element_box["height"] <= footer_box["y"] + footer_box["height"]
        ):
            is_page_footer = False

    return is_page_footer


def _is_page_num(element):
    """Find if element is a page number.

    Parameters
    ----------
    element : Unstructured Element dictionary
        The element to be checked.

    Returns
    -------
    bool
        Returns True if the element is likely to be a page number, False otherwise.

    """
    is_page_num = False

    # Capture page layout
    page_width = element["metadata"]["coordinates"]["layout_width"]
    page_height = element["metadata"]["coordinates"]["layout_height"]

    # Define page corners as ratio % of the page corners
    ratio = 0.15
    inv_ratio = 1 - ratio

    left_top = {"x": 0, "y": 0, "height": page_height * ratio, "width": page_width * ratio}

    right_top = {"x": page_width * inv_ratio, "y": 0, "height": page_height * ratio, "width": page_width * ratio}

    left_bottom = {"x": 0, "y": page_height * inv_ratio, "height": page_height * ratio, "width": page_width * ratio}

    right_bottom = {
        "x": page_width * inv_ratio,
        "y": page_height * inv_ratio,
        "height": page_height * ratio,
        "width": page_width * ratio,
    }

    corner_boxes = [left_top, right_top, left_bottom, right_bottom]

    # Define element coordinates
    coords = element["metadata"]["coordinates"]["points"]
    element_box = {
        "x": coords[0][0],
        "y": coords[0][1],
        "height": coords[1][1] - coords[0][1],
        "width": coords[2][0] - coords[0][0],
    }

    # Find if element is within the corner boxes
    for corner_box in corner_boxes:
        # If top-left inner box corner is inside the bounding box
        if corner_box["x"] <= element_box["x"] and corner_box["y"] <= element_box["y"]:
            # If bottom-right inner box corner is inside the bounding box
            if (
                element_box["x"] + element_box["width"] <= corner_box["x"] + corner_box["width"]
                and element_box["y"] + element_box["height"] <= corner_box["y"] + corner_box["height"]
            ):
                is_page_num = True

    return is_page_num


def _is_main_title(element):
    """Find if element is the document main title.

    Parameters
    ----------
    element : Unstructured Element dictionary
        The element to be checked.

    Returns
    -------
    bool
        Returns True if the element is likely to be the document main title, False otherwise.

    """
    is_main_title = False

    element_class = element["metadata"]["element_class"]
    element_class_plus = element["metadata"]["element_class_plus"]

    if element_class_plus not in ["PageHeader", "PageNumber"] and element_class == "Title":
        is_main_title = True

    return is_main_title


def _is_section_header(element):
    """Find if element is a section header.

    Parameters
    ----------
    element : Unstructured Element dictionary
        The element to be checked.

    Returns
    -------
    list of bool
        Returns a list of boolean values, with the check if the element is a section header, section sub header or
        section sub sub header, respectively.

    """
    is_section_header1 = False
    is_section_header2 = False
    is_section_header3 = False

    element_text = element["text"]

    if element["metadata"]["element_class"] == "Title":
        header1_search = re.search("^[\d]?[\d]?[\d][.|-]?[ ]", element_text)
        header2_search = re.search("^[\d]?[\d]?[\d][.|-][\d]?[\d]?[\d][ ]", element_text)
        header3_search = re.search("^[\d]?[\d]?[\d][.|-][\d]?[\d]?[\d][.|-][\d]?[\d]?[\d][ ]", element_text)

        if header1_search is not None:
            is_section_header1 = True

        if header2_search is not None:
            is_section_header2 = True

        if header3_search is not None:
            is_section_header3 = True

        if element_text.count(" ") >= len(element_text) * 0.3:
            # Another possible option to identify OCR mistakes is to look for the existence of
            # "detection_class_prob" in the element metadata keys. Need more examples to test if there's support for this.
            # Unlikely a real text, most likely a missinterpretation from OCR
            is_section_header1 = False
            is_section_header2 = False
            is_section_header3 = False

    return [is_section_header1, is_section_header2, is_section_header3]


def _is_toc_who_guidelines(element):
    """Flag the handpicked ToC in WHO guidelines."""
    is_toc = False

    # List of element ids marked as ToC
    who_toc_ids = [
        "9d10b3a29e65406fd4eb6883857e0896",
        "b07504548130c0da23c5df2cdddfb81c",
        "46b918e9da8a4b7011cbdc29299704ff",
        "e5b363547471c969519bc29ee904e01f",
        "3b42427dd6704abbb208d68c12438d0a",
        "9437e3545e10fc7e83ff901f91b786d2",
        "ee3f22aba4f9ec79cb777689adcd2d8e",
        "56d9364953d899fc79638e3490968420",
        "7497e147a049414098be3097b005a859",
        "a80d3614859e4ebb360b9ea4bc9865de",
        "deec0413363ab98594a1a9d907870c29",
        "25fd6aebf2c174149625d4941412a3c3",
        "510b3e1d56012fa37e631646d9bb8d7d",
        "f0175e3752d8e769974243dafbc649f1",
    ]

    if element["element_id"] in who_toc_ids:
        is_toc = True

    return is_toc


def add_element_context(elements, stop_title_search_after):
    """Based on a list of Unstructured elements, add more context to each element class. Search for elements that are page numbers, enumerated section titles or page headers and footers.

    Parameters
    ----------
    elements : list of Unstructured Elements
        The list of elements to look for more specific classes.

    stop_title_search_after: int
        The number of elements to check for existence of a main title. The search will start on element 0 and stop at
        element number stop_title_search_after.

    Returns
    -------
    list of Unstructured Elements dictionaries
        Returns a list of Unstructured Elements converted to dictionaries. The returned elements will have two
        more metadata objects: element_class, which will reflect the class identified by Unstructured and
        element_class_plus with the results of the class finetuning.

    """
    main_title = ""
    elements_plus = []

    for el_indx, unst_element in enumerate(elements):
        element = unst_element.to_dict()
        # Save element type to it's metadata
        element["metadata"]["element_class"] = element["type"]
        # Initialize the finetuned element class
        element["metadata"]["element_class_plus"] = ""

        # Identify if element is document main title
        if main_title == "":  # Still haven't found title
            if _is_main_title(element) and el_indx < stop_title_search_after:
                # Titles are generally one of the first elements in the document
                # If it's position is larger than stop_title_search_after is unlikely that it's a main title
                main_title = element["text"]
                element["metadata"]["element_class_plus"] = "MainTitle"

        # Identify if element is page header
        if _is_page_header(element):
            element["metadata"]["element_class_plus"] = "PageHeader"

        # Identify if element is section header
        if _is_section_header(element)[0]:
            element["metadata"]["element_class_plus"] = "SectionHeader1"

        if _is_section_header(element)[1]:
            element["metadata"]["element_class_plus"] = "SectionHeader2"

        if _is_section_header(element)[2]:
            element["metadata"]["element_class_plus"] = "SectionHeader3"

        # Identify if element is page number
        if _is_page_num(element):
            element["metadata"]["element_class_plus"] = "PageNumber"

        # Identify if element is page footer
        if _is_page_footer(element):
            element["metadata"]["element_class_plus"] = "PageFooter"

        # Identify if table has a title preceding it to add it to element metadata
        if (el_indx > 0) and (element["type"] == "Table") and (elements[el_indx - 1].to_dict()["type"] == "Title"):
            element["metadata"]["title"] = elements[el_indx - 1].text

        # Flag for possible Table of Contents element
        if _is_toc_who_guidelines(element):
            element["metadata"]["is_toc"] = True
        else:
            element["metadata"]["is_toc"] = False

        elements_plus.append(element)

    return elements_plus


def find_main_title(elements_plus, stop_title_search_after):
    """Based on a list of Unstructured elements, search for main title.

    Parameters
    ----------
    elements : list of Unstructured Elements
        The list of elements to be searched.

    stop_title_search_after: int
        The number of elements to check for existence of a main title. The search will start on element 0 and stop at
        element number stop_title_search_after.

    Returns
    -------
    a str with the title. Empty str if no title was found.
    """
    for el_indx, element in enumerate(elements_plus):
        if element["metadata"]["element_class_plus"] == "MainTitle":
            return element["text"]
        if el_indx > stop_title_search_after:
            # Main title won't be this far from 1st element, stop looking
            return ""

    return ""


def add_main_title(elements_plus, stop_title_search_after):
    """Add main title metadata to all elements on list.

    Parameters
    ----------
    elements : list of Unstructured Elements with element_class_plus attributes
        The list of elements to add main title metadata.

    stop_title_search_after: int
        The number of elements to check for existence of a main title. The search will start on element 0
        and stop at element number stop_title_search_after.

    Returns
    -------
    list of Elements as dictionaries
        Returns a list of dictionaries with a main_title attribute in the metadata.

    """
    main_title = find_main_title(elements_plus, stop_title_search_after)
    for element in elements_plus:
        element["metadata"]["doc_main_title"] = main_title

    return elements_plus


def add_section_metadata(elements_plus):
    """Add section header metadata to elements.

    Parameters
    ----------
    elements_plus : list of Unstructured Elements with element_class_plus attributes
        The list of elements to add metadata.

    Returns
    -------
    list of Elements as dictionaries
        Returns a list of Elements (tables, text etc) with current section metadata.

    """
    # Initialize vairables
    current_header1 = ""  # To be stored in the element metadata
    current_header2 = ""  # To be stored in the element metadata
    current_header3 = ""  # To be stored in the element metadata

    for el_indx, element in enumerate(elements_plus):
        # Search for headers
        if element["metadata"]["element_class_plus"] == "SectionHeader1":
            # Found header
            current_header1 = element["text"]
            # Reset sub section headers:
            current_header2 = ""
            current_header3 = ""

        if element["metadata"]["element_class_plus"] == "SectionHeader2":
            # Found subheader
            current_header2 = element["text"]
            # Reset sub section headers:
            current_header3 = ""

        if element["metadata"]["element_class_plus"] == "SectionHeader3":
            # Found subheader
            current_header3 = element["text"]

        element["metadata"]["section_header1"] = current_header1
        element["metadata"]["section_header2"] = current_header2
        element["metadata"]["section_header3"] = current_header3

    return elements_plus


def remove_toc_elements(elements_plus):
    """Remvoes all elements flagged as Table of Contents from a list of elements.

    Parameters
    ----------
    elements_plus : list of Unstructured Elements with element_class_plus attributes
        The list of elements to search from.

    Returns
    -------
    list of Elements as dictionaries
        Returns a list of Elements without those flagged as Table of Contents.

    """
    filtered_elements = []
    for element in elements_plus:
        if element["metadata"]["is_toc"] is False:
            filtered_elements.append(element)

    return filtered_elements
