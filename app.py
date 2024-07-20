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
from twilio.twiml.messaging_response import MessagingResponse


load_dotenv()

app = Flask(__name__)

# Twilio account credentials
account_sid = os.getenv("my_account_sid")
auth_token = os.getenv("my_auth_token")

# Creating the Twilio client
client = Client(account_sid, auth_token)

# Updating the webhook URL for the phone
phone_number_sid = os.getenv("my_phone_number_sid")
webhook_url = "https://9d80-2a0d-5600-44-4000-00-9fb.ngrok-free.app/ivr"
phone_number = client.incoming_phone_numbers(phone_number_sid).fetch()
phone_number.update(voice_url=webhook_url, voice_method='POST')


# Whatsapp credentials
whatsapp_number = "whatsapp:+14155238886"
whatsapp_account_sid = os.getenv("whatsapp_sid")
whatsapp_auth_token = os.getenv("whatsapp_auth")
whatsapp_client = Client(whatsapp_account_sid, whatsapp_auth_token)
WHATSAPP_USER_PHONE_NUMBER = "whatsapp:+263773344079"


# OpenAI API key
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

embeddings_path = "emdeddings_dataset.csv"
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"
df = pd.read_csv(embeddings_path)
df['embedding'] = df['embedding'].apply(ast.literal_eval)

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
    
    # Log ranked strings and relatedness scores
    app.logger.info(f"Ranked strings: {strings}")
    app.logger.info(f"Relatedness scores: {relatednesses}")

    introduction = 'Use the below information from the Star International. Answer as a virtual assistance for the company. Try your best to answer all the questions using the provided information. If the answer cannot be found in the info, write "I could not find a satisfactory answer for your question. Please, contact our number, on 0 7 7 8 0 4 0 4 9 7 3 or visit our website (www.starinternational.co.zw) for more information."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nINFORMATION FOR Star International:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    
    final_message = message + question
    
    # Log the final constructed message
    app.logger.info(f"Constructed message: {final_message}")

    return final_message

def ask(
    query: str, df: pd.DataFrame = df,
    model: str = GPT_MODEL, token_budget: int = 4096 - 500, print_message: bool = False,
) -> str:
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about Star International and persuade customers to use the transporting services. Be friendly and empathetic."},
        {"role": "user", "content": message},
    ]
    response = openai_client.chat.completions.create(model=model, messages=messages, temperature=0)
    response_message = response.choices[0].message.content
    
    # Log the response from OpenAI
    app.logger.info(f"Response from OpenAI: {response_message}")
    
    return response_message

# Counter to track the number of interactions
interaction_counter = 0

# Allowed maximum number of interactions before ending the call
MAX_INTERACTIONS = 15
USER_PHONE_NUMBER = "+263773344079"

@app.route('/', methods=['GET'])
def home():
    return "Welcome to Tau's IVR"

@app.route('/ivr', methods=['POST'])
def handle_ivr():
    global interaction_counter
    response = VoiceResponse()
    speech_input = request.values.get('SpeechResult', '').lower()
    
    # Log the captured speech input
    app.logger.info(f"Captured Speech Input: {speech_input}")

    # Greeting message
    if interaction_counter == 0:
        response.say("Hi, this is Tau from Star International. How can I help you today?", voice='Polly.Gregory-Neural')
        # Prompt the caller to respond
        response.gather(input='speech', timeout=3, speechTimeout='auto', action='/ivr')
        interaction_counter += 1
        return str(response)

    # Use the ask function to get the response from the NLP model
    response_text = ask(speech_input, df)

    # Log the response from OpenAI
    app.logger.info(f"Response from OpenAI: {response_text}")

    # Convert text to speech using Twilio's TTS
    response.say(response_text, voice='Polly.Gregory-Neural')
    interaction_counter += 1
    
    # Checking if the maximum number of interactions has been reached
    if interaction_counter >= MAX_INTERACTIONS:
        response.say("Thank you for calling Star International. Have a great day!")
        response.hangup()
        interaction_counter = 0  # Reset counter after call ends
    else:
        # Follow-up marketing question
        response.say("By the way, do you have any loads for us to carry?", voice='Polly.Gregory-Neural')
        response.gather(input='speech', timeout=3, speechTimeout='auto', action='/ivr')
    
    return str(response)

@app.route('/call-user', methods=['POST'])
def call_user():
    try:
        call = client.calls.create(
            url=webhook_url, to=USER_PHONE_NUMBER, from_='+16467590558'
        )

        # Logging call initiation
        app.logger.info(f"Call initiated. Call SID: {call.sid}")
        
        # Creating a new VoiceResponse for the call initiation
        call_response = VoiceResponse()
        
        # Greeting message when the call is initiated
        call_response.say("Hi, this is Tau from Star International, how are you doing today?", voice='Polly.Gregory-Neural')
        
        # Follow-up question
        call_response.say("Great!S I just wanted to ask. Do you have any loads for us?", voice='Polly.Gregory-Neural')
        
        # Return the response
        return jsonify({"message": "Call initiated", "call_sid": call.sid}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/send-whatsapp', methods=['POST'])
def send_whatsapp():
    try:
        message = client.messages.create(
            body="Hi, this is Tau from Star International. How are you doing today?",
            from_=whatsapp_number,
            to=WHATSAPP_USER_PHONE_NUMBER
        )

        # Logging WhatsApp message initiation
        app.logger.info(f"WhatsApp message sent. Message SID: {message.sid}")
        
        return jsonify({"message": "WhatsApp message sent", "message_sid": message.sid}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/whatsapp', methods=['POST'])
def handle_whatsapp():
    incoming_msg = request.values.get('Body', '').lower()
    response = MessagingResponse()
    message = response.message()
    
    # Log the incoming WhatsApp message
    app.logger.info(f"Incoming WhatsApp message: {incoming_msg}")

    # Use the ask function to get the response from the NLP model
    response_text = ask(incoming_msg, df)

    # Log the response from OpenAI
    app.logger.info(f"Response from OpenAI: {response_text}")

    # Send the response back to the user on WhatsApp
    message.body(response_text)
    
    # Follow-up marketing message
    message.body("By the way, do you have any loads for us to carry?")
    
    return str(response)


if __name__ == '__main__':
    app.run(port=8000, debug=True)





