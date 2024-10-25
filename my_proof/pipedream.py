
import requests


def pipedream_print(msg: str):
    url = "https://eoipxj5xgopiaqq.m.pipedream.net/"
    headers = {'Content-Type': 'application/json'}
    data = {
        "message": msg
    }

    response = requests.post(url, json=data, headers=headers)

    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text}")
