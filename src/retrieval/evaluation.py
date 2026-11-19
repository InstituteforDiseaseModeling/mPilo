from langchain.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.retrievers import ContextualCompressionRetriever, BM25Retriever, EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.embeddings.openai import OpenAIEmbeddings
import pandas as pd
from sagemaker.experiments import load_run
import mlflow

# Batch size for loading documents
MAX_BATCH_SIZE = 100


def create_evaluation_df(retrieval_qa):
    """Convert evaluation data into the DataFrame format required by MLFlow.

    Parameters
    ----------
    retrieval_qa : dict
        Dictionary with questions as keys and expected answers as values
    Returns
    -------
    Pandas DataFrame
        Evaluation DataFrame expected by MLFlow
    """
    ## Create ID mapping for queries and for corpus
    all_questions = []
    all_answers = []
    for question, answers in retrieval_qa.items():
        all_questions.append(question)
        if type(answers) is not str:
            for answer in answers:
                all_answers.append(answer)
        else:
            all_answers.append(answers)

    quest_no_dup = list(set(all_questions))
    ans_no_dup = list(set(all_answers))

    qid = ["q" + str(i) for i in range(len(quest_no_dup))]
    aid = ["a" + str(i) for i in range(len(ans_no_dup))]

    queries = {qid[i]: quest_no_dup[i] for i in range(len(qid))}
    corpus = {aid[i]: ans_no_dup[i] for i in range(len(aid))}

    ## Map original questions to newly created IDs
    reference_docs = {}
    for question, answers in retrieval_qa.items():
        idx = list(queries.keys())[list(queries.values()).index(question)]
        if type(answers) is not str:
            idy = []
            for answer in answers:
                idy.append(list(corpus.keys())[list(corpus.values()).index(answer)])

        else:
            idy = [list(corpus.keys())[list(corpus.values()).index(answers)]]

        reference_docs[str(idx)] = idy

    question_df = pd.DataFrame()
    question_df["qid"] = reference_docs.keys()
    question_df["question"] = question_df.qid.apply(lambda x: queries[x])
    question_df["aid"] = question_df.qid.apply(lambda x: reference_docs[x])
    question_df["answers"] = question_df.question.apply(
        lambda x: list(retrieval_qa[x]) if type(retrieval_qa[x]) is set else [retrieval_qa[x]]
    )
    return question_df


def evaluate_retriever(retriever, experiment_name, model_name, sm_session, question_df):
    """Run retrieval evaluation using MLFlow and save results to SageMaker Experiments.

    Parameters
    ----------
    retriever : RetrievalEvaluation
        The retrieval model to be evaluated
    experiment_name : str
        Name of the experiment to be saved on SageMaker Experiments
    model_name : str
        Name of the model for SageMaker Experiments run name
    sm_session: SageMaker Session
        SageMaker Session instance
    question_df: Pandas DataFrame
        DataFrame produced by create_evaluation_df

    Returns
    -------
    dict
        Evaluation results from MLFlow
    """

    def retriever_model_function(question_df, retriever=retriever):
        return question_df["question"].apply(retriever.retrieve_doc_ids)

    metrics = ["ndcg_at_10", "recall_at_1", "recall_at_2", "recall_at_3"]

    # Run names can only be alphanumeric with hyphens
    run_name = model_name.replace("/", "-").replace(".", "")

    with load_run(experiment_name=experiment_name, run_name=run_name, sagemaker_session=sm_session) as sm_run:
        with mlflow.start_run():
            evaluate_results = mlflow.evaluate(
                model=retriever_model_function,
                data=question_df,
                model_type="retriever",
                targets="answers",
                evaluators="default",
                extra_metrics=[
                    mlflow.metrics.recall_at_k(1),
                    mlflow.metrics.recall_at_k(2),
                    mlflow.metrics.recall_at_k(3),
                    mlflow.metrics.ndcg_at_k(10),
                ],
            )
        for metric in metrics:
            sm_run.log_metric(name=metric, value=evaluate_results.metrics[f"{metric}/mean"])

    return evaluate_results.metrics


class RetrievalEvaluation:
    """
    A class for evaluating document retrievers.

    Parameters
    ----------
    documents : list
        List of documents for creating the vector store.
    embedding : LangChain Embedding
        The embedding model to be used for encoding the documents.
    k : int, optional
        The number of nearest neighbors to retrieve, by default 10.
    """

    def __init__(
        self, documents, embedding, k=10  # e.g. HuggingFaceEmbeddings(model_name=model_name))
    ):  # Must be >= the largest k required for metrics
        """Initialize class."""
        self.vectorstore = FAISS.from_documents(documents, embedding=embedding)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

    def retrieve_doc_ids(self, question):
        """
        Retrieve document IDs relevant to the given question.

        Parameters
        ----------
        question : str
            The question for which relevant document IDs are to be retrieved.

        Returns
        -------
        list
            List of document IDs relevant to the question.
        """
        docs = self.retriever.get_relevant_documents(question)
        # Here docs are identified by their content. We could also use page number for fuzzier matching.
        doc_ids = [doc.page_content for doc in docs]
        return doc_ids

    def retrieve_llm_content(self, question):
        """
        Retrieve document IDs relevant to the given question.

        Parameters
        ----------
        question : str
            The question for which relevant document IDs are to be retrieved.

        Returns
        -------
        list
            List of document IDs relevant to the question.
        """
        docs = self.retriever.get_relevant_documents(question)
        # Here docs are identified by their content. We could also use page number for fuzzier matching.
        doc_ids = [doc.metadata["llm_content"] for doc in docs]
        return doc_ids

    def retrieve_page_numbers(self, question):
        """
        Retrieve deduplicated page numbers relevant to the given question.

        Parameters
        ----------
        question : str
            The question for which relevant document IDs are to be retrieved.

        Returns
        -------
        list
            List of document IDs relevant to the question.
        """
        docs = self.retriever.get_relevant_documents(question)
        # Here docs are identified by their content. We could also use page number for fuzzier matching.
        doc_ids = list(set([str(doc.metadata["start_page"]) for doc in docs]))
        return doc_ids


class Small2BigEvaluation(RetrievalEvaluation):
    """
    A class for evaluating small2big document retrievers.

    Parameters
    ----------
    documents : list
        List of documents for creating the vector store.
    embedding : LangChain Embedding
        The embedding model to be used for encoding the documents.
    k : int, optional
        The number of nearest neighbors to retrieve, by default 10.
    child_split_characters : int
        Number of characters to split child documents on.
    """

    def __init__(
        self, documents, embedding, child_split_characters, k=10  # e.g. HuggingFaceEmbeddings(model_name=model_name))
    ):  # Must be >= the largest k required for metrics
        """Initialize class."""
        # The vectorstore to use to index the child chunks
        self.child_vectorstore = Chroma(collection_name="full_documents", embedding_function=embedding)

        # The storage layer for the parent documents
        self.parent_store = InMemoryStore()

        child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_split_characters)

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.child_vectorstore,
            docstore=self.parent_store,
            child_splitter=child_splitter,
            search_kwargs={"k": k},
        )

        batches = [documents[i : i + MAX_BATCH_SIZE] for i in range(0, len(documents), MAX_BATCH_SIZE)]

        for batch in batches:
            self.retriever.add_documents(batch, ids=None)


class BGERerankerEvaluation(RetrievalEvaluation):
    """
    A class for evaluating document retriever with BGEReranker.

    Parameters
    ----------
    documents : list
        List of documents for creating the vector store.
    embedding : LangChain Embedding
        The embedding model to be used for encoding the documents.
    k : int, optional
        The number of nearest neighbors to retrieve, by default 10.
    """

    def __init__(
        self, documents, embedding, reranker, k=10  # e.g. HuggingFaceEmbeddings(model_name=model_name))
    ):  # Must be >= the largest k required for metrics
        """Initialize class."""
        self.vectorstore = FAISS.from_documents(documents, embedding=embedding)
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        self.retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)


class MultiQueryRetrievalEvaluation(RetrievalEvaluation):
    """
    A class for evaluating multiquery retrieval.

    Parameters
    ----------
    documents : list
        List of documents for creating the vector store.
    embedding : LangChain Embedding
        The embedding model to be used for encoding the documents.
    llm : LangChain LLM
        The LLM to use for query generation
    k : int, optional
        The number of nearest neighbors to retrieve, by default 10.
    """

    def __init__(
        self, documents, embedding, llm, k=10  # e.g. HuggingFaceEmbeddings(model_name=model_name))
    ):  # Must be >= the largest k required for metrics
        """Initialize class."""
        self.vectorstore = FAISS.from_documents(documents, embedding=embedding)
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        self.retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)


class HybridRetrievalEvaluation(RetrievalEvaluation):
    """
    A class for evaluating hybrid retrieval.

    Parameters
    ----------
    documents : list
        List of documents for creating the vector store.
    embedding : LangChain Embedding
        The embedding model to be used for encoding the documents.
    bm25_weight : float
        The proportion of scoring to weight to BM25
    k : int, optional
        The number of nearest neighbors to retrieve, by default 10.
    """

    def __init__(
        self, documents, embedding, bm25_weight, k=10  # e.g. HuggingFaceEmbeddings(model_name=model_name))
    ):  # Must be >= the largest k required for metrics
        """Initialize class."""
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k
        self.vectorstore = FAISS.from_documents(documents, embedding=embedding)
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, base_retriever], weights=[bm25_weight, 1 - bm25_weight]
        )


class HybridRetrievalOpenAIEvaluation(RetrievalEvaluation):
    """
    A class for evaluating hybrid retrieval.

    Parameters
    ----------
    documents : list
        List of documents for creating the vector store.
    embedding : LangChain Embedding
        The embedding model to be used for encoding the documents.
    bm25_weight : float
        The proportion of scoring to weight to BM25
    k : int, optional
        The number of nearest neighbors to retrieve, by default 10.
    """

    def __init__(
        self, documents, embedding, openai_weight, k=10  # e.g. HuggingFaceEmbeddings(model_name=model_name))
    ):  # Must be >= the largest k required for metrics
        """Initialize class."""
        openai_vectorstore = FAISS.from_documents(documents, embedding=OpenAIEmbeddings())
        openai_retriever = openai_vectorstore.as_retriever(search_kwargs={"k": k})
        self.vectorstore = FAISS.from_documents(documents, embedding=embedding)
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        self.retriever = EnsembleRetriever(
            retrievers=[openai_retriever, base_retriever], weights=[openai_weight, 1 - openai_weight]
        )
