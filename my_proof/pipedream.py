
import sys
import requests


def pipedream_print(msg: str):
    url = "https://eoipxj5xgopiaqq.m.pipedream.net/"
    headers = {'Content-Type': 'application/json'}
    data = {
        "message": msg
    }
    response = requests.post(url, json=data, headers=headers)
    return response


if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "No message provided"
    pipedream_print(msg)
