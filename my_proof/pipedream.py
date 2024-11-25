
import sys
import requests


def pipedream_print(msg: str):
    """
    For external logging, swap out 'url' for your pipedream endpoint and
    use pipedream_print() like you would logging.info() or print()
    """
    url = "https://some-url.m.pipedream.net/"  # Add your custom pipedream url here
    headers = {'Content-Type': 'application/json'}
    data = {
        "message": msg
    }
    response = requests.post(url, json=data, headers=headers)
    return response


if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "No message provided"
    pipedream_print(msg)
