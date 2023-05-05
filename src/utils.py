import json


def read_tokens(filename="api_keys.json"):
    with open(filename) as f:
        return json.load(f)
