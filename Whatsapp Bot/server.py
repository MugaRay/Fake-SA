import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import flask
import numpy as np
import keras
from pywa import WhatsApp
from pywa.types import Message
from pywa import filters as fil


flask_app = flask.Flask(__name__)


bert = keras.saving.load_model("BERT models/bert_ww_merge.keras")


wa =  WhatsApp(
    phone_id = os.environ.get("PHONE_ID"),
    server = flask_app,
    verify_token = os.environ.get("VERIFY_TOKEN"),
    token= os.environ.get("WHATAPP_TOKEN"),
)


# helper function
def convert_to_logs(n):
    """converts the BERT output from logirthms to percentages"""
    logs = np.exp(n) / np.sum(np.exp(n), axis=1, keepdims=True)
    return logs[0][0], logs[0][1]


def predict_msg(message):
    """ predicts the the message and generate a string accordingly"""
    predict = bert.predict([message])
    fake, _ = convert_to_logs(predict)
    msg = ""
    if fake > 0.8:
        msg = "high likelihood of being disinformation. This means the information may be very unreliable."
    elif fake > 0.6:
        msg = "moderate likelihood. It’s advisable to double-check this information."
    else:
        msg = "low likelihood. While it seems mostly accurate, verifying is always a good practice."
    return f"Your message has a {msg} of being disinformation. Please always verify information from trusted sources."


# routing functions
@wa.on_message(fil.startswith("hello", "hi", ignore_case=True))
def hello(_: WhatsApp, msg: Message):
    """Hello message for the router"""
    welcome_msg = f"""
    👋 Welcome, {msg.from_user.name}, to the Fake News Bot!

    If you want to check the likelihood of some information being disinformation, just type "check:" followed by the information you want to verify.

    Example:
    check: Is FakeNewsBot real or is it extraterrestrial? 👽
    I’m here to help you navigate the truth! What would you like to check today?
    """
    msg.reply(welcome_msg)


@wa.on_message(fil.startswith("check:", ignore_case=True))
def predict_infromatation(_:WhatsApp, msg:Message):
    """
    This function purely checks if the input text is fake news or not
    """
    resp = predict_msg(msg.text)
    msg.reply(resp)


@wa.on_message(fil.not_(fil.matches("hi", "hello", "check:", ignore_case=True)))
def helper_func(_ : WhatsApp, msg : Message):
    """For helper functions"""
    help_msg = """
    🛠️ How to Use the Fake News Bot

    Welcome to the Fake News Bot! Here’s how to get the most out of this service:

    Check for Disinformation:
    To verify information, simply type "check:" followed by the statement or claim you want to investigate.
    Example:
    check: Is climate change real?
    """
    msg.reply(help_msg)

# flask web server
flask_app.run(port=3000)