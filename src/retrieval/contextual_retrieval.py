"""Retrieve chunks of a document contextually e.g. with section names."""
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import Tool
from langchain.tools.retriever import RetrieverInput
import functools
import os
import subprocess

from src.config import IS_LOCAL, MODELS_S3_URI

script_dir = os.path.dirname(os.path.abspath(__file__))
RETRIEVAL_EMBEDDING = os.path.join(script_dir, "../../gpt4-finetuned-retriever/")

# Load retrieval model from S3

if not IS_LOCAL and MODELS_S3_URI:
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            f"{MODELS_S3_URI}/gpt4-finetuned-retriever/",
            RETRIEVAL_EMBEDDING,
            "--delete",
        ]
    )


def contextual_retrieval_tool(
    documents,
    name="search_who_document",
    description="Search 'Consolidated guidelines on HIV prevention, testing, treatment, service delivery and monitoring' document.",
    k=2,
    embedding_model=HuggingFaceEmbeddings(model_name=RETRIEVAL_EMBEDDING),
    save_locally=False,
    vectorstore_name="vectorstore",
):
    """Tool to retrieve chunks of a document contextually e.g. with section names.

    Parameters
    ----------
    documents : list
        List of Document objects.
    name : str
        Name of tool for LLM prompt.
    description : str
        Description of tool for LLM prompt.
    k : int, optional
        Top number of closest documents to return, by default 2
    embedding_model : langchain.embeddings.Embeddings, optional
        Model with which to embed documents, by default HuggingFaceEmbeddings(model_name=RETRIEVAL_EMBEDDING)
    save_locally : bool, optional
        Boolean flag to save the vectors locally, by default False
    vectorstore_name : str, optional
        Name of the local vector store, by default name is vectorstore

    Returns
    -------
    langchain.tools.Tool
        Langchain retrieval tool.
    """
    if os.path.isdir(vectorstore_name):
        # Check if existing vectorestore exists in local
        vectorstore = FAISS.load_local(vectorstore_name, embedding_model, allow_dangerous_deserialization=True)
    else:
        # Create vectorstore.
        vectorstore = FAISS.from_documents(documents, embedding=embedding_model)
    # Save vectorstore locally
    if save_locally:
        vectorstore.save_local(vectorstore_name)
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    # Assemble Tool.
    tool = Tool(
        name=name,
        description=description,
        # Use functools.partial so that function can be called with only query argument, as expected by tool.
        func=functools.partial(formatted_docs, retriever=retriever),
        coroutine=functools.partial(formatted_docs, retriever=retriever),
        args_schema=RetrieverInput,
    )
    return tool


def formatted_docs(query, retriever):
    """Chooses relevant metadata from document and formats in a single string to send to LLM.

    Parameters
    ----------
    query : str
        Query to send to retriever.
    retriever : langchain_core.retrievers.BaseRetriever
        Retrieval model.

    Returns
    -------
    str
        String representation of returned documents.
    """
    docs = retriever.get_relevant_documents(query)
    doc_string_representations = [doc_string_representation(doc) for doc in docs]
    return "\n".join(doc_string_representations)


def doc_string_representation(doc):
    """Chooses relevant metadata from document and formats in a single string to send to LLM.

    Parameters
    ----------
    doc : langchain.schema.document.Document
        Relevent document returned by retrieval model.

    Returns
    -------
    str
        String representation of document.
    """
    # Page(s) logic
    if doc.metadata["start_page"] == doc.metadata["end_page"]:
        page = f"Page: {doc.metadata['start_page']}"
    else:
        page = f"Pages: {doc.metadata['start_page']}-{doc.metadata['end_page']}"
    section_list = [doc.metadata["section_header1"], doc.metadata["section_header2"], doc.metadata["section_header3"]]
    # If the document is from a table with a title, add it.
    if "title" in doc.metadata.keys():
        section_list.append(doc.metadata["title"])
    # Remove empty sections
    section_list = filter(lambda x: len(x) > 0, section_list)
    # Join sections with arrow to show hierarchy.
    sections = "->".join(section_list)
    # Final assembly
    return f"""[Section: {sections} {page} Relevant excerpt: '{doc.metadata["llm_content"]}']"""
