from src.config import get_secrets, OPENAI_MODEL, REDIS_SERVER_ADDRESS, REDIS_PORT
from src.persistent_memory.memory_manager import MemoryManager
from src.server.redis_utils import is_connected

from langchain_openai import ChatOpenAI
import os
import redis
import sys

# Expect 2 arguments:
# sys.argv[1] is the user name or "all" to run on all users
user_name = sys.argv[1]
# sys.argv[2] is a bool to use translation layers or not
translate = sys.argv[2]
# sys.argv[3] is a str representing the user language
user_language = sys.argv[3]
# sys.argv[4] is the number of interactions to retain
num_inter_retain = sys.argv[4]


# Connect to Redis
rds = redis.Redis(host=REDIS_SERVER_ADDRESS, port=REDIS_PORT, decode_responses=True)

_ = get_secrets()
llm = ChatOpenAI(temperature=0, model_name=OPENAI_MODEL)

# Instantiate MemoryManager
memory_manager = MemoryManager(redis_connection=rds, llm=llm, translate=translate, user_language=user_language)

if is_connected(rds):
    if user_name == "all":
        result = memory_manager.update_all_users(retain_k=num_inter_retain)
    else:
        result = memory_manager.update_user_details(user=user_name, retain_k=num_inter_retain)
else:
    print("Not connected to Redis")
    result = False


print(result)
