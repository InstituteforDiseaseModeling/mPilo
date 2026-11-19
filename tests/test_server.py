import pytest
from fastapi.testclient import TestClient
from src.server.twilio_server import app

client = TestClient(app)


@pytest.fixture
def mock_agent_response(mocker):
    """Mock the response from the OpenAI API to save cost."""
    mock_response = {"output": "Hi, how can I help you?"}
    mocker.patch(
        "src.chatbot.adaptive_agent.AdaptiveFunctionsAgent.to_executor"
    ).return_value.invoke.return_value = mock_response
    return mocker


@pytest.fixture
def mock_redis_db(mocker, redisdb):
    """Provide a temporary redisdb."""
    mocker.patch("src.server.twilio_server.redis.Redis").return_value = redisdb
    return mocker


class TestServer:
    """Test the Twilio server API."""

    def test_chat_with_clear_memory_command(self, mock_redis_db, mock_agent_response):
        """Test that the command 'clear memory' clears the memory."""
        # Add something to memory first
        client.post("/hook", data={"From": "whatsapp+1234567890", "Body": "Hi there"})
        # Clear memory command
        response = client.post("/hook", data={"From": "whatsapp+1234567890", "Body": "clear memory"})
        assert response.status_code == 200
        assert "Memory cleared" in response.text

    def test_chat_with_free_text(self, mock_redis_db, mock_agent_response):
        """Test that free text input returns the expected mocked response."""
        response = client.post("/hook", data={"From": "whatsapp+1234567890", "Body": "Hi there"})
        assert response.status_code == 200
        assert "Hi, how can I help you?" in response.text
