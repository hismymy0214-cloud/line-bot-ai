# app.py
import os

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from bot_core import build_reply

app = Flask(__name__)

# 從環境變數讀取 LINE Bot 的金鑰
CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")

if not CHANNEL_ACCESS_TOKEN or not CHANNEL_SECRET:
    raise RuntimeError(
        "請先在環境變數 LINE_CHANNEL_ACCESS_TOKEN 與 LINE_CHANNEL_SECRET 設定你的 LINE Bot 金鑰。"
    )

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)


@app.route("/", methods=["GET"])
def index():
    return "LINE bot is running."


@app.route("/callback", methods=["POST"])
def callback():
    # LINE 簽章
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    user_text = event.message.text.strip()
    reply_text = build_reply(user_text)

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )


if __name__ == "__main__":
    # 本機測試用；在 Render 上會用 gunicorn 啟動，不會走到這裡
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
