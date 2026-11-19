import json
import re
import warnings

SHORT_TERM_MEMORY_KEY = "short_term_memory"
LONG_TERM_MEMORY_KEY = "long_term_memory"
USER_DETAILS_KEY = "user_details"
USER_SEGMENT_PROBS_KEY = "user_segment_probs"


def is_connected(client, silent=False):
    """Check Redis connection."""
    try:
        client.keys()
        connected = True

    except Exception as ex:
        connected = False
        template = "Could not communicate with Redis server. Exception {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        if not silent:
            warnings.warn(message)

    return connected


def ingest_to_db(client, prefix, user, _dict):
    """Convert dict into JSON and store in REDIS db at key: prefix/user."""
    client.set(f"{prefix}/{user}", json.dumps(_dict))
    return True


def retrieve_from_db(client, prefix, user):
    """Retrieve JSON from key prefix/user of REDIS db and return dict."""
    _json = client.get(f"{prefix}/{user}")
    return json.loads(_json)


def delete_key_from_user(client, prefix, user):
    """Delete a key / prefix from a specific user."""
    client.delete(f"{prefix}/{user}")
    return True


def delete_user(client, user):
    """Delete all keys / prefixes associated with a user."""
    all_redis_keys = client.keys()
    all_resp = [client.delete(rkey) for rkey in all_redis_keys if re.search(f"/{user}$", rkey) is not None]
    return all(all_resp)


def get_all_users(client):
    """Get a list of all users registered in redis."""
    all_redis_keys = client.keys()
    all_users = [rkey.split("/")[1] for rkey in all_redis_keys if re.search("/", rkey) is not None]
    all_users_dedup = list(dict.fromkeys(all_users))
    return all_users_dedup


def get_all_users_short_mem(client):
    """Retrieve all users with short term memories."""
    all_keys = client.keys()
    all_redis_keys = [key.decode() for key in all_keys if type(key) == bytes] + [
        key for key in all_keys if type(key) == str
    ]

    st_mem = [
        rkey.split("/")[1] for rkey in all_redis_keys if re.search(f"^{SHORT_TERM_MEMORY_KEY}/", rkey) is not None
    ]
    return st_mem
