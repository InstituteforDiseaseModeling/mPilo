from src.chatbot.segmentation import (
    find_non_None,
    freq_prob,
    cluster_prob,
    equal_probs,
    chat_history_prob,
    new_input_prob,
    highest_prob,
    highest_prob_above_threshold,
    adjust_probs,
    CENTROIDS,
)
from pandas.core.frame import DataFrame
from langchain_community.chat_models.fake import FakeMessagesListChatModel
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


def get_dict(x):
    """Help to build a fake classifier chain."""
    return x["input"]


class TestSegmentation:
    """Tests for the segmentation aux functions."""

    def test_find_non_None(self):
        """Test the function to find elements that are not None."""
        assert find_non_None(["a", "b", "c", None]) == [0, 1, 2]
        assert find_non_None([None, None]) == []

    def test_freq_prob(self):
        """Test the frequency-based probability generator."""
        probs_no_neutral = freq_prob(neutral_weight=0)
        assert type(probs_no_neutral) == DataFrame
        assert all(x in probs_no_neutral.columns.values.tolist() for x in ["segment", "prob", "evidence"])
        assert probs_no_neutral.loc[probs_no_neutral["segment"] == "Opaque"].prob.item() == 0.0

        probs_neutral = freq_prob(neutral_weight=0.5)
        assert type(probs_neutral) == DataFrame
        assert all(x in probs_neutral.columns.values.tolist() for x in ["segment", "prob", "evidence"])
        assert probs_neutral.loc[probs_neutral["segment"] == "Opaque"].prob.item() == 0.5

    def test_cluster_prob(self):
        """Test the cluster-based probability generator."""
        cluster_probs = cluster_prob(centroids=CENTROIDS, patient_values=[1, 1, 0, 0, 5, 4, 4, 3, 2, 1])
        assert type(cluster_probs) == DataFrame
        assert all(x in cluster_probs.columns.values.tolist() for x in ["segment", "prob", "evidence"])
        assert highest_prob(cluster_probs) == "Teal"

    def test_equal_probs(self):
        """Test the equal-based probability generator."""
        equal_prob_df = equal_probs()
        assert type(equal_prob_df) == DataFrame
        assert all(x in equal_prob_df.columns.values.tolist() for x in ["segment", "prob", "evidence"])

    def test_chat_history_prob(self, mr_rose_conversation_buffer):
        """Test the history-based probability generator."""
        fake_classifier_llm = FakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content='{\n    "Teal": 1,\n    "Rose": 0,\n    "Green": 0,\n    "Blue": 0,\n    "Gray": 0,\n    "Opaque": 0\n}'
                )
            ]
        )

        fake_classifier_chain = RunnableLambda(get_dict) | fake_classifier_llm

        empty_conversation_buffer = ConversationBufferMemory(memory_key="chat_history", output_key="output")
        empty_probs = chat_history_prob(
            conversation_buffer=empty_conversation_buffer, llm_classifier=fake_classifier_chain
        )
        assert type(empty_probs) == DataFrame
        assert all(x in empty_probs.columns.values.tolist() for x in ["segment", "prob", "evidence"])

        filled_conversation_buffer = mr_rose_conversation_buffer
        filled_probs = chat_history_prob(
            conversation_buffer=filled_conversation_buffer, llm_classifier=fake_classifier_chain
        )
        assert type(filled_probs) == DataFrame
        assert all(x in filled_probs.columns.values.tolist() for x in ["segment", "prob", "evidence"])

    def test_new_input_probs(self, mr_rose_conversation_buffer):
        """Test the chat-based probability generator."""
        fake_classifier_llm = FakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content='{\n    "Teal": 1,\n    "Rose": 0,\n    "Green": 0,\n    "Blue": 0,\n    "Gray": 0,\n    "Opaque": 0\n}'
                )
            ]
        )

        fake_classifier_chain = RunnableLambda(get_dict) | fake_classifier_llm

        empty_conversation_buffer = ConversationBufferMemory(memory_key="chat_history", output_key="output")
        new_input = "Hello World!"
        empty_probs = new_input_prob(
            conversation_buffer=empty_conversation_buffer, new_input=new_input, llm_classifier=fake_classifier_chain
        )
        assert type(empty_probs) == DataFrame
        assert all(x in empty_probs.columns.values.tolist() for x in ["segment", "prob", "evidence"])

        filled_conversation_buffer = mr_rose_conversation_buffer
        new_input = "What is HIV?"
        filled_probs = new_input_prob(
            conversation_buffer=filled_conversation_buffer, new_input=new_input, llm_classifier=fake_classifier_chain
        )
        assert type(filled_probs) == DataFrame
        assert all(x in filled_probs.columns.values.tolist() for x in ["segment", "prob", "evidence"])

    def test_highest_prob(self):
        """Test the function to retrun segment with highest probability."""
        assert highest_prob(freq_prob(neutral_weight=0)) == "Rose"
        assert highest_prob(freq_prob(neutral_weight=0.5)) == "Opaque"

    def test_highest_prob_above_threshold(self):
        """Test the function to retrun segment with highest probability above a threshold."""
        assert highest_prob_above_threshold(freq_prob(neutral_weight=0), null_class="Opaque", threshold=0.5) == "Opaque"

        assert highest_prob_above_threshold(freq_prob(neutral_weight=0), null_class="Opaque", threshold=0.2) == "Rose"

        assert (
            highest_prob_above_threshold(freq_prob(neutral_weight=0.2), null_class="Opaque", threshold=0.2) == "Opaque"
        )

        assert highest_prob_above_threshold(freq_prob(neutral_weight=0.1), null_class="Opaque", threshold=0.2) == "Rose"

    def test_adjust_probs(self):
        """Test the function that adjusts probabilities."""
        priori = equal_probs()
        posteriori = freq_prob(neutral_weight=0)
        posteriori.rename(columns={"evidence": "new_evidence", "prob": "new_prob"}, inplace=True)
        new_probs = adjust_probs(priori, posteriori)
        assert type(new_probs) == DataFrame
        assert all(x in new_probs.columns.values.tolist() for x in ["segment", "prob", "evidence"])
