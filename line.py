import os
import requests
import json


def send_to_masaumi(inp):
    # LINE Messaging APIのURL
    url = 'https://api.line.me/v2/bot/message/push'

    # LINE Messaging APIのアクセストークン
    token = os.getenv('LINE_ACCESS_TOKEN')

    header = {'Content-Type': 'application/json',
              'Authorization': 'Bearer {}'.format(token)}

    # メッセージを送信する相手のID
    _id = 'U1d52a9596a5219a8bf9782749006fe06'

    if type(inp) is str:
        # テキストメッセージの場合
        texts = [{"type": "text",
                  "text": inp}]
        body = {"to": _id,
                "messages": texts}
    else:
        # Flex Messageの場合
        body = {"to": _id,
                "messages": inp}

    # LINE Messaging APIにリクエストを送信
    res = requests.post(url, json.dumps(body), headers=header)

    return res
