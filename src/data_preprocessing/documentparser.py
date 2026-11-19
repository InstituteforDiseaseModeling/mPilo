"""Preprocessing of unstructured data."""
from io import StringIO
import pandas as pd
from unstructured.partition.pdf import partition_pdf
from src.data_preprocessing.tableocr import table_element_llm_html, summarise_table
from src.data_preprocessing.element_contextualize import add_element_context, add_section_metadata, remove_toc_elements
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import Document
import asyncio
import nest_asyncio
import re
import warnings

nest_asyncio.apply()


class DocumentParser:
    """Parser of unstructured data.

    Parameters
    ----------
    document_character_limit : int
        Maximum number of characters allowed in a document.
    document_chunk_overlap : int
        Number of characters overlapping between chunks.

    """

    def __init__(self, document_character_limit, document_chunk_overlap=20, text_splitter="recursive"):
        """Instantiate class."""
        self.document_character_limit = document_character_limit
        self.document_chunk_overlap = document_chunk_overlap
        self.text_splitter = text_splitter

    def from_pdf(self, pdf_filepath, gpt4v_tables=False, who_document=False):
        """Parse a PDF file and return a list of Document objects.

        Parameters
        ----------
        pdf_filepath : str
            Path to the PDF file to parse.
        gpt4v_tables : bool
            Option to generate higher quality HTML representations of tables using GPT4V.

        Returns
        -------
        list of Document
            List of Langchain Document objects representing the parsed PDF.

        """
        pdf_elements = self._partition_pdf(pdf_filepath)

        if gpt4v_tables:
            # Replace html content of pdf_elements using GPT4V
            pdf_elements = self._gpt4v_tables(pdf_elements, pdf_filepath)

        pdf_docs = self.from_elements(pdf_elements, who_document=who_document)

        return pdf_docs

    def from_elements(self, elements, who_document=False):
        """Parse a list of Element objects and return a list of Document objects.

        Parameters
        ----------
        elements : list
            List of Unstructured Element objects.

        Returns
        -------
        list of Document
            List of Langchain Document objects representing the parsed PDF.

        """
        # Add element class contextualization to the metadata
        elements_plus_context = add_element_context(elements, stop_title_search_after=4)

        # Add section headers to element metadata
        elements_with_section = add_section_metadata(elements_plus_context)

        # Remove Table of Contents from the WHO guidelines
        if who_document:
            elements_with_section = remove_toc_elements(elements_with_section)

        # Parse tables
        table_docs = self._chunk_tables(elements_with_section)

        # Parse text
        text_in_sections = self._merge_text_section(elements_with_section)
        text_docs = self._chunk_text(text_in_sections, document_character_limit=self.document_character_limit)

        pdf_docs = table_docs + text_docs

        return pdf_docs

    def _partition_pdf(self, pdf_filepath):
        """Partition a PDF file into its constituent elements.

        Parameters
        ----------
        pdf_filepath : str
            Path to the PDF file to partition.

        Returns
        -------
        list
            List of Element objects.

        """
        pdf_elements = partition_pdf(
            filename=pdf_filepath,
            # Unstructured first finds embedded image blocks
            extract_images_in_pdf=False,
            # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
            # Titles are any sub-section of the document
            infer_table_structure=True,
        )

        return pdf_elements

    def _gpt4v_tables(self, pdf_elements, pdf_filepath):
        """Generate higher quality HTML representations of tables using GPT4V.

        Parameters
        ----------
        pdf_elements : list
            List of PDF Element objects.

        Returns
        -------
        pdf_elements : list
            List of PDF Element objects with updated HTML representations.

        """
        loop = asyncio.get_event_loop()
        pdf_elements = loop.run_until_complete(table_element_llm_html(pdf_elements, pdf_filepath))

        return pdf_elements

    def _chunk_tables(self, pdf_elements, add_summary=False, summary_option=0):
        """Chunk PDF elements into tables and return a list of Document objects.

        Parameters
        ----------
        pdf_elements : list
            List of PDF Element objects to chunk.
        add_summary : bool
            boolean flag to have an option to add Table summary
        summary_option : int (0 or 1)
            0: Table raw text passed as input to summary module
            1: Table HTML passed as input to summary module

        Returns
        -------
        list of Document
            List of Langchain Document objects representing the chunked tables.

        """
        # Filter pdf_elements to isolate tables. Included in this method in case non-table elements need to be added or position of tables relative to text is taken into account.
        table_elements = [element for element in pdf_elements if "Table" in element["type"]]

        table_docs = []

        for element in table_elements:
            # Capture the closest section header from metadata
            if element["metadata"]["section_header3"] == "":
                if element["metadata"]["section_header2"] == "":
                    latest_header = element["metadata"]["section_header1"]
                else:
                    latest_header = element["metadata"]["section_header2"]
            else:
                latest_header = element["metadata"]["section_header3"]

            try:
                chunk_table_docs = self._chunk_table(
                    element["metadata"]["text_as_html"], element["metadata"]["page_number"], latest_header
                )
            except Exception as e:
                warnings.warn(f"No HTML found for Table element id {element['element_id']}. Error {e}")
                continue

            # to check if summary has to be added or not
            if add_summary:
                # invoke the llm to generate summary
                summary = (
                    summarise_table(element["metadata"]["text_as_html"])
                    if summary_option
                    else summarise_table(element["text"])
                )

            for doc in chunk_table_docs:
                doc.metadata["element_class"] = element["type"]
                doc.metadata["section_header1"] = element["metadata"]["section_header1"]
                doc.metadata["section_header2"] = element["metadata"]["section_header2"]
                doc.metadata["section_header3"] = element["metadata"]["section_header3"]
                if add_summary:
                    doc.metadata["table_summary"] = summary

            table_docs += chunk_table_docs

        return table_docs

    def _chunk_rows(self, df, start, header_len=0):
        """Check Chunk size against document chunk size hard limit and return the index of the dataframe.

        Parameters
        ----------
        df : Dataframe
            Pandas Dataframe object with content of table.
        start: integer
            Row from where to start chunking.

        Returns
        -------
        index
            Index of dataframe representing the row till which the chunking should happen.

        """
        j = start
        while j < len(df) and len(df.iloc[start : j + 1, :].to_html()) + header_len <= self.document_character_limit:
            j += 1

        if j > len(df):
            return j - 1
        return j

    def _chunk_table(self, table_html, page_number, section_header=""):
        """Chunk a table HTML string and return a Document object.

        Parameters
        ----------
        table_html : str
            HTML string representing the table to chunk.
        page_number : int
            Page number of PDF
        section_header : str
            Closest section header title

        Returns
        -------
        Document
            List of Langchain Document objects representing the chunked table.

        """
        df_list = pd.read_html(StringIO(table_html), flavor="lxml")
        df = df_list[0]

        # extracting  title of table
        seq = "<" + "caption" + ">(.*?)</" + "caption" + ">"
        tab_title = re.findall(seq, table_html)

        # check if length of the table dataframe is less than the length of document level chunk size
        if len(df.to_html() + section_header) <= self.document_character_limit:
            # check if title is present in the table html
            if tab_title:
                return [
                    Document(
                        page_content=section_header + df.to_string(index=False).replace("   ", ""),
                        metadata={
                            "title": tab_title[0],
                            "llm_content": df.to_html(),
                            "start_page": page_number,
                            "end_page": page_number,
                        },
                    )
                ]
            else:
                return [
                    Document(
                        page_content=section_header + df.to_string(index=False).replace("   ", ""),
                        metadata={
                            "llm_content": df.to_html(),
                            "start_page": page_number,
                            "end_page": page_number,
                        },
                    )
                ]

        # chunk table based on document level chunk size rowwise
        chunk_row_idx_list = [list(range(i, self._chunk_rows(df, i, len(section_header)))) for i in range(len(df))]

        df_chunks = [df.iloc[chunk_row_idx, :] for chunk_row_idx in chunk_row_idx_list]

        # check if title is present in the table html
        if tab_title:
            return [
                Document(
                    page_content=section_header + df_chunk.to_string(index=False).replace("   ", ""),
                    metadata={
                        "title": tab_title[0],
                        "llm_content": df_chunk.to_html(),
                        "start_page": page_number,
                        "end_page": page_number,
                    },
                )
                for df_chunk in df_chunks
            ]
        else:
            return [
                Document(
                    page_content=section_header + df_chunk.to_string(index=False).replace("   ", ""),
                    metadata={
                        "llm_content": df_chunk.to_html(),
                        "start_page": page_number,
                        "end_page": page_number,
                    },
                )
                for df_chunk in df_chunks
            ]

    def _merge_text_section(self, elements_with_sections):
        """Merge text elements based on the section they belong.

        Parameters
        ----------
        elements_with_sections : list of Unstructured Elements with section attributes.
            The list of elements to be merged.

        Returns
        -------
        list of Document
            Returns a list of Langchain Documents with merged text.

        """
        # The following classes will not be merged into text chunks
        avoid_class = ["Table"]
        avoid_class_plus = [
            "MainTitle",
            "PageHeader",
            "PageFooter",
            "PageNumber",
            "SectionHeader1",
            "SectionHeader2",
            "SectionHeader3",
        ]

        # Initialize vairables
        chunks = []  # Will append merged texts
        chunk = ""  # To store the text within each chunk
        current_header1 = ""  # To be stored in the element metadata
        current_header2 = ""  # To be stored in the element metadata
        current_header3 = ""  # To be stored in the element metadata
        start_page = 1  # To be stored in the element metadata
        end_page = 1  # To be stored in the element metadata

        for el_indx, element in enumerate(elements_with_sections):
            # Search for headers
            if element["metadata"]["element_class_plus"] in ["SectionHeader1", "SectionHeader2", "SectionHeader3"]:
                # Found new header, output previous text chunk and start a new one
                if len(chunk) > 0:  # Output previous chunk if it's length is greater than 0
                    text_chunk = Document(
                        page_content=chunk,
                        metadata={
                            "element_class": "NarrativeTextSection",
                            "start_page": start_page,
                            "end_page": end_page,
                            "section_header1": current_header1,
                            "section_header2": current_header2,
                            "section_header3": current_header3,
                        },
                    )
                    chunks.append(text_chunk)

                # Start new chunk
                chunk = ""
                current_header1 = element["metadata"]["section_header1"]
                current_header2 = element["metadata"]["section_header2"]
                current_header3 = element["metadata"]["section_header3"]
                start_page = element["metadata"]["page_number"]
                # Skip to next iteration
                continue

            # If element is not from an unwanted class, merge it to the text chunk
            if (
                element["metadata"]["element_class"] not in avoid_class
                and element["metadata"]["element_class_plus"] not in avoid_class_plus
            ):
                chunk = chunk + "\n" + str(element["text"])
                # Overwrite chunk end page before new iteration
                end_page = element["metadata"]["page_number"]

        # Loop through elements ends here ----------------

        # Append last chunk, if it's not empty
        if len(chunk) > 0:
            text_chunk = Document(
                page_content=chunk,
                metadata={
                    "element_class": "NarrativeTextSection",
                    "start_page": start_page,
                    "end_page": end_page,
                    "section_header1": current_header1,
                    "section_header2": current_header2,
                    "section_header3": current_header3,
                },
            )
            chunks.append(text_chunk)

        return chunks

    def _chunk_text(self, merged_text, document_character_limit):
        """Divide text sections into pieces that have size smaller than document_character_limit.

        Parameters
        ----------
        merged_text : list Document
            The list of Langchain Documents

        document_character_limit : int
            Hard limit to number of tokens allowed in Document llm_content section.


        Returns
        -------
        list of Document
            Returns Langchain documents with character count equal or less than document_character_limit.

        """
        # Apply Langchain document splitter based on token limit
        if self.text_splitter == "recursive":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=document_character_limit,
                chunk_overlap=self.document_chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
        elif self.text_splitter == "character":
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=document_character_limit,
                chunk_overlap=self.document_chunk_overlap,
                length_function=len,
            )

        splitted_docs = text_splitter.split_documents(merged_text)

        # Add llm content metadata
        for doc in splitted_docs:
            doc.metadata["llm_content"] = doc.page_content

        return splitted_docs
