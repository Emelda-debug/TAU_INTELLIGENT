import os
import ast
import pandas as pd
import re
import json
from scipy import spatial
from flask import Flask, request, jsonify
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Twilio account credentials
account_sid = os.getenv("my_account_sid")
auth_token = os.getenv("my_auth_token")

# Creating the Twilio client
client = Client(account_sid, auth_token)

# Updating the webhook URL for the phone
phone_number_sid = os.getenv("my_phone_number_sid")
webhook_url = "https://e6b8-2c0f-2a80-1231-3410-e427-f159-2198-d5f6.ngrok-free.app/ivr"
phone_number = client.incoming_phone_numbers(phone_number_sid).fetch()
phone_number.update(voice_url=webhook_url, voice_method='POST')

# OpenAI API key
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    )

embeddings_path = "emdeddings_dataset.csv"
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"
df = pd.read_csv(embeddings_path)
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    query_embedding_response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL, input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str, df: pd.DataFrame, model: str, token_budget: int
) -> str:
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below information from the Star International. Answer as a virtual assistance for the company. Try your best to answer all the questions using the provided information. If the answer cannot be found in the info, write "I could not find a satisfactory answer for your question. Please, contact our number, on +263 7780404973 or visit our website (www.starinternational.co.zw) for more information."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nINFORMATION FOR Star International:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    return message + question

def ask(
    query: str, df: pd.DataFrame = df,
    model: str = GPT_MODEL, token_budget: int = 4096 - 500, print_message: bool = False,
) -> str:
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about Star International."},
        {"role": "user", "content": message},
    ]
    response = openai_client.chat.completions.create(model=model, messages=messages, temperature=0)
    response_message = response.choices[0].message.content
    return response_message

# Counter to track the number of interactions
interaction_counter = 0

# Allowed maximum number of interactions before ending the call
MAX_INTERACTIONS = 3
USER_PHONE_NUMBER = "+263718240384"

@app.route('/', methods=['GET'])
def home():
    return "Welcome to Tau's IVR"

@app.route('/ivr', methods=['POST'])
def handle_ivr():
    global interaction_counter
    response = VoiceResponse()
    speech_input = request.values.get('SpeechResult', '').lower()
    
    # Use the ask function to get the response from the NLP model
    response_text = ask(speech_input, df)
    
    # Convert text to speech using Twilio's TTS
    response.say(response_text, voice='alice')
    interaction_counter += 1
    
    # Checking if the maximum number of interactions has been reached
    if interaction_counter >= MAX_INTERACTIONS:
        response.hangup()
    else:
        # Continuing the IVR flow
        response.gather(input='speech', timeout=3, speechTimeout='auto', action='/ivr')
    
    return str(response)

@app.route('/call-user', methods=['POST'])
def call_user():
    try:
        call = client.calls.create(
            url=webhook_url, to=USER_PHONE_NUMBER, from_='+15017991650'
        )
        return jsonify({"message": "Call initiated", "call_sid": call.sid}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=8000, debug=True)