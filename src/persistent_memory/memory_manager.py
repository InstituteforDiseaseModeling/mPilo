from src.persistent_memory.persistent_memory import PersistentMemory
from src.persistent_memory.prompt import EXTRACTOR_PROMPT, MERGER_PROMPT
from src.server.redis_utils import (
    ingest_to_db,
    retrieve_from_db,
    get_all_users_short_mem,
    SHORT_TERM_MEMORY_KEY,
    LONG_TERM_MEMORY_KEY,
    USER_DETAILS_KEY,
)

from langchain.schema import messages_from_dict, messages_to_dict


class MemoryManager:
    """Manage persistent memories and short / long term chat history.

    Parameters
    ----------
    redis_connection : redis.client.Redis
        A redis client connection.
    llm :
        The LLM to be used for summarizing and merging the important features extracted.
    translate : bool
        Whether translation of the chat memory is necessary. (Default is False).
    user_language : str
        The string representing the language of the chat memory. (Default is None).
    """

    def __init__(self, redis_connection, llm, translate=False, user_language=None):
        self.redis_connection = redis_connection
        self.short_term_memory_key = SHORT_TERM_MEMORY_KEY
        self.long_term_memory_key = LONG_TERM_MEMORY_KEY
        self.user_details_key = USER_DETAILS_KEY
        self.persistent_memory = PersistentMemory(
            llm=llm, extractor_prompt=EXTRACTOR_PROMPT, merger_prompt=MERGER_PROMPT
        )
        self.translate = translate
        self.user_language = user_language

    def update_user_details(self, user, retain_k=0, silent=True):
        """Run the pipeline to update user details.

        Parameters
        ----------
        user : str
            The user name string.
        retain_k : int
            The number of interactions to keep in the short term memory after running the persistent memory extraction. (Default is 0).
        silent : bool
            If the update is silent. If not, print a message at the beginning of each step. (Default is True).

        Returns
        -------
        bool
            Returns True when the process is completed.
        """
        # Step 1: Read any short term memories and previously extracted user details
        if not silent:
            print(f"== Updating memories for user {user}...")
            print("1. Reading short term memories and user details...")
        if self.redis_connection.exists(self.short_term_memory_key + "/" + user):
            short_term_memory = messages_from_dict(
                retrieve_from_db(client=self.redis_connection, prefix=self.short_term_memory_key, user=user)
            )
            if short_term_memory == "" or len(short_term_memory) == 0:
                print(f"No new memories, skipping user {user}.")
                return True
        else:
            print(f"No new memories, skipping user {user}.")
            return True

        if self.redis_connection.exists(self.user_details_key + "/" + user):
            user_details = retrieve_from_db(client=self.redis_connection, prefix=self.user_details_key, user=user)
        else:
            user_details = "- No info."

        # Step 2: Apply persistent memory generation
        if not silent:
            print("2. Applying persistent memory extraction...")

        new_user_details = self.persistent_memory.extract_and_merge_features(
            chat_messages=short_term_memory,
            older_features=user_details,
            translate=self.translate,
            user_language=self.user_language,
        )

        # Step 3: Write new persistent memory to redis
        if not silent:
            print("3. Writing new persistent memory...")
        ingest_to_db(client=self.redis_connection, prefix=self.user_details_key, user=user, _dict=new_user_details)

        # Step 4: Append short term memory to long term memory
        if not silent:
            print("4. Appending short term memory to long term memory...")
        if self.redis_connection.exists(self.long_term_memory_key + "/" + user):
            long_term_memory = messages_from_dict(
                retrieve_from_db(client=self.redis_connection, prefix=self.long_term_memory_key, user=user)
            )
        else:
            long_term_memory = []

        if retain_k > 10:
            retain_k = 10
            print("NOTE: Interactions to retain capped at 10")
        elif retain_k < 0:
            retain_k = 0
            print("NOTE: Negative interactions to retain reset to 0")

        if retain_k == 0:
            retained_memories = ""
            not_retained_memories = short_term_memory
        else:
            retain_index = -2 * retain_k
            retained_memories = messages_to_dict(short_term_memory[retain_index : len(short_term_memory)])
            not_retained_memories = short_term_memory[:retain_index]

        long_term_memory = long_term_memory + not_retained_memories
        ingest_to_db(
            client=self.redis_connection,
            prefix=self.long_term_memory_key,
            user=user,
            _dict=messages_to_dict(long_term_memory),
        )

        # Step 5: Clear short term memory
        if not silent:
            print("5. Clearing short term memory...")

        ingest_to_db(
            client=self.redis_connection, prefix=self.short_term_memory_key, user=user, _dict=retained_memories
        )

        if not silent:
            print(f"== Memory updated for user {user}.")
        return True

    def update_all_users(self, retain_k=0, silent=True):
        """Run the pipeline to update all user details.

        Parameters
        ----------
        silent : bool
            If the update is silent. If not, print a message at the beginning of each step. (Default is True).
        retain_k : int
            The number of interactions to keep in the short term memory after running the persistent memory extraction. (Default is 0).

        Returns
        -------
        bool
            Returns True when the process is completed.
        """
        users_to_update = get_all_users_short_mem(client=self.redis_connection)

        if len(users_to_update) > 0:
            update_res = [
                self.update_user_details(user=user, retain_k=retain_k, silent=silent) for user in users_to_update
            ]
            return all(update_res)
        else:
            if not silent:
                print("No users to update.")
            return True
