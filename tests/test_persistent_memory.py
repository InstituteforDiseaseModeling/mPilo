from src.persistent_memory.persistent_memory import PersistentMemory
from src.persistent_memory.prompt import EXTRACTOR_PROMPT, MERGER_PROMPT
from src.persistent_memory.memory_manager import MemoryManager
from src.server.redis_utils import (
    ingest_to_db,
    retrieve_from_db,
    SHORT_TERM_MEMORY_KEY,
    LONG_TERM_MEMORY_KEY,
    USER_DETAILS_KEY,
)

from langchain_community.chat_models.fake import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

import pytest


@pytest.fixture
def mock_llm():
    """Create a mock LLM to use in the tests."""
    fake_llm = FakeMessagesListChatModel(
        responses=[AIMessage(content="{Old list:\n- No info. New features:\n-No info.}")]
    )
    return fake_llm


class TestPersistentMemory:
    """Unit tests for the PersistentMemory class."""

    def test_extract_and_merge_features(self, mock_llm, mr_rose_conversation_buffer):
        """Test the extraction and merging of user features."""
        persistent_memory = PersistentMemory(
            llm=mock_llm, extractor_prompt=EXTRACTOR_PROMPT, merger_prompt=MERGER_PROMPT
        )

        extracted_details = persistent_memory.extract_and_merge_features(
            chat_messages=mr_rose_conversation_buffer.chat_memory.messages,
        )

        assert type(extracted_details) is str


class TestMemoryManager:
    """Unit tests for the MemoryManager class."""

    def test_update_user_details(self, mock_llm, redisdb):
        """Test updating single user details."""
        users = ["fake_user1", "fake_user2"]
        for user in users:
            ingest_to_db(
                client=redisdb,
                prefix=SHORT_TERM_MEMORY_KEY,
                user=user,
                _dict=[
                    {
                        "type": "human",
                        "data": {
                            "content": "Hello, Coach.",
                            "additional_kwargs": {},
                            "type": "human",
                            "name": None,
                            "example": False,
                        },
                    },
                    {
                        "type": "ai",
                        "data": {
                            "content": "Hello. How are you?",
                            "additional_kwargs": {},
                            "type": "ai",
                            "name": None,
                            "example": False,
                        },
                    },
                ],
            )
            ingest_to_db(
                client=redisdb,
                prefix=LONG_TERM_MEMORY_KEY,
                user=user,
                _dict=[
                    {
                        "type": "human",
                        "data": {
                            "content": "Hello, Coach.",
                            "additional_kwargs": {},
                            "type": "human",
                            "name": None,
                            "example": False,
                        },
                    },
                    {
                        "type": "ai",
                        "data": {
                            "content": "Hello. How are you?",
                            "additional_kwargs": {},
                            "type": "ai",
                            "name": None,
                            "example": False,
                        },
                    },
                ],
            )
            ingest_to_db(
                client=redisdb,
                prefix=USER_DETAILS_KEY,
                user=user,
                _dict="- Patient name is Jon.\\n- Patient is 23 years old.",
            )

        memory_manager = MemoryManager(redis_connection=redisdb, llm=mock_llm)
        # Test on a user with memories
        updated_details = memory_manager.update_user_details(user="fake_user1")
        assert updated_details

        # Test on a user without memories
        updated_details = memory_manager.update_user_details(user="fake_user3")
        assert updated_details

        # Test memories are getting retained correctly
        retain_k = 2
        lt_mem_before = retrieve_from_db(client=redisdb, prefix=LONG_TERM_MEMORY_KEY, user="fake_user2")
        st_mem_before = retrieve_from_db(client=redisdb, prefix=SHORT_TERM_MEMORY_KEY, user="fake_user2")
        expected_len_lt = len(lt_mem_before) + max(len(st_mem_before) - (retain_k * 2), 0)

        updated_details = memory_manager.update_user_details(user="fake_user2", retain_k=retain_k)

        lt_mem_after = retrieve_from_db(client=redisdb, prefix=LONG_TERM_MEMORY_KEY, user="fake_user2")
        st_mem_after = retrieve_from_db(client=redisdb, prefix=SHORT_TERM_MEMORY_KEY, user="fake_user2")

        assert len(lt_mem_after) == expected_len_lt
        assert len(st_mem_after) == min(len(st_mem_before), retain_k * 2)

    def test_update_all_users(self, mock_llm, redisdb):
        """Test updating multiple user details."""
        users = ["fake_user1", "fake_user2"]
        for user in users:
            ingest_to_db(
                client=redisdb,
                prefix=SHORT_TERM_MEMORY_KEY,
                user=user,
                _dict=[
                    {
                        "type": "human",
                        "data": {
                            "content": "Hello, Coach.",
                            "additional_kwargs": {},
                            "type": "human",
                            "name": None,
                            "example": False,
                        },
                    },
                    {
                        "type": "ai",
                        "data": {
                            "content": "Hello. How are you?",
                            "additional_kwargs": {},
                            "type": "ai",
                            "name": None,
                            "example": False,
                        },
                    },
                ],
            )
            ingest_to_db(
                client=redisdb,
                prefix=LONG_TERM_MEMORY_KEY,
                user=user,
                _dict=[
                    {
                        "type": "human",
                        "data": {
                            "content": "Hello, Coach.",
                            "additional_kwargs": {},
                            "type": "human",
                            "name": None,
                            "example": False,
                        },
                    },
                    {
                        "type": "ai",
                        "data": {
                            "content": "Hello. How are you?",
                            "additional_kwargs": {},
                            "type": "ai",
                            "name": None,
                            "example": False,
                        },
                    },
                ],
            )
            ingest_to_db(
                client=redisdb,
                prefix=USER_DETAILS_KEY,
                user=user,
                _dict="- Patient name is Jon.\\n- Patient is 23 years old.",
            )

        memory_manager = MemoryManager(redis_connection=redisdb, llm=mock_llm)
        updated_details = memory_manager.update_all_users()
        assert updated_details
