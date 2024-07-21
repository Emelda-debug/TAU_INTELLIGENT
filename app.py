import os
import ast
import json
import pandas as pd
from scipy import spatial
from flask import Flask, request, jsonify
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from twilio.twiml.messaging_response import MessagingResponse
from textblob import TextBlob

load_dotenv()

app = Flask(__name__)

# Twilio account credentials
account_sid = os.getenv("my_account_sid")
auth_token = os.getenv("my_auth_token")
client = Client(account_sid, auth_token)

# Update the webhook URL for the phone
phone_number_sid = os.getenv("my_phone_number_sid")
webhook_url = "https://53ea-2605-6440-4000-e000-00-2856.ngrok-free.app/ivr"
phone_number = client.incoming_phone_numbers(phone_number_sid).fetch()
phone_number.update(voice_url=webhook_url, voice_method='POST')

# WhatsApp credentials
whatsapp_number = "whatsapp:+14155238886"
whatsapp_account_sid = os.getenv("whatsapp_sid")
whatsapp_auth_token = os.getenv("whatsapp_auth")
whatsapp_client = Client(whatsapp_account_sid, whatsapp_auth_token)
WHATSAPP_USER_PHONE_NUMBER = "whatsapp:+263773344079"

# OpenAI API key
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embeddings_path = "DATASET/emdeddings_dataset.csv"
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
        model=EMBEDDING_MODEL, input=query
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
    
    introduction = 'Use the below information from the Star International. Answer as a virtual assistant for the company. Try your best to answer all the questions using the provided information. If the answer cannot be found in the info, write "Sorry, I can not fully answer that, instead let me refer you to my colleague, who will reach out shortly, if they delay please, contact our number, 0 7 7 8 0 4 0 4 9 7 3 or visit our website (www.starinternational.co.zw) for more information."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nINFORMATION FOR Star International:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    
    final_message = message + question
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
    return response_message

# Global variables for IVR and WhatsApp handling
interaction_counter = 0
MAX_INTERACTIONS = 15
greeted = False
asked_about_wellbeing = False
asked_about_services = False
gathered_feedback = False

def load_existing_customers():
    with open('customers.json', 'r') as file:
        data = json.load(file)
    return data["existing_customers"]

existing_customers = load_existing_customers()

def get_customer_status(phone_number):
    return "existing" if phone_number in existing_customers else "new"

@app.route('/', methods=['GET'])
def home():
    return "Welcome to Tau's IVR"

@app.route('/ivr', methods=['POST'])
def handle_ivr():
    global interaction_counter, greeted, asked_about_wellbeing, asked_about_services, gathered_feedback
    response = VoiceResponse()
    speech_input = request.values.get('SpeechResult', '').lower()
    
    # Handle initial greeting
    if not greeted:
        phone_number = request.values.get('From', '')
        customer_status = get_customer_status(phone_number)
        
        if customer_status == "new":
            response.say("Hi, this is Tau from Star International. How are you doing today?", voice='Polly.Gregory-Neural')
        else:
            response.say("Hi, this is Tau from Star International. How can I assist you today?", voice='Polly.Gregory-Neural')

        response.gather(input='speech', timeout=3, speechTimeout='auto', action='/ivr')
        greeted = True
        interaction_counter += 1
        return str(response)

    if greeted and not asked_about_wellbeing:
        # Ask about well-being after initial greeting
        sentiment = TextBlob(speech_input).sentiment.polarity
        if sentiment < 0:
            response.say("I'm sorry to hear that you're not feeling well. What's wrong?", voice='Polly.Gregory-Neural')
            asked_about_wellbeing = True
            interaction_counter += 1
            response.gather(input='speech', timeout=3, speechTimeout='auto', action='/ivr')
            return str(response)
        else:
            response.say("Glad to hear you're doing well! By the way, do you have any loads for us to carry?", voice='Polly.Gregory-Neural')
            asked_about_wellbeing = True
            asked_about_services = True
            response.gather(input='speech', timeout=3, speechTimeout='auto', action='/ivr')
            interaction_counter += 1
            return str(response)

    if asked_about_wellbeing and not asked_about_services:
        # Handle response to load inquiry
        if "yes" in speech_input:
            response.say("Great! Could you please provide more information about the load?", voice='Polly.Gregory-Neural')
            asked_about_services = True
            gathered_feedback = True
        elif "no" in speech_input:
            response.say("Thank you for your time. If you need any assistance, feel free to reach out to us anytime. Have a great day!", voice='Polly.Gregory-Neural')
            response.hangup()
            interaction_counter = 0  # Reset counter after interaction ends
            greeted = False  # Reset greeting flag
            asked_about_wellbeing = False
            asked_about_services = False
            gathered_feedback = False
        else:
            response.say("Sorry, I didn't understand that. Can you please tell me if you have any loads for us to carry?", voice='Polly.Gregory-Neural')
        return str(response)
    
    if gathered_feedback:
        # Follow-up if needed
        response.say("Thank you for the information. Is there anything else we can help you with?", voice='Polly.Gregory-Neural')
        response.gather(input='speech', timeout=3, speechTimeout='auto', action='/ivr')
        return str(response)

    # Use the ask function to get the response from the NLP model
    response_text = ask(speech_input, df)
    response.say(response_text, voice='Polly.Gregory-Neural')
    interaction_counter += 1

    # Check if the maximum number of interactions has been reached
    if interaction_counter >= MAX_INTERACTIONS:
        response.say("Thank you for reaching out to Star International. We are here whenever you need us. Please feel free to reach out anytime. Have a great day!", voice='Polly.Gregory-Neural')
        response.hangup()
        interaction_counter = 0  # Reset counter after interaction ends
        greeted = False  # Reset greeting flag
        asked_about_wellbeing = False
        asked_about_services = False
        gathered_feedback = False
    elif not asked_about_services:
        # Follow-up marketing message
        response.say("By the way, do you have any loads for us to carry?", voice='Polly.Gregory-Neural')
        response.gather(input='speech', timeout=3, speechTimeout='auto', action='/ivr')

    return str(response)

@app.route('/send-whatsapp', methods=['POST'])
def send_whatsapp():
    global greeted, asked_about_wellbeing, asked_about_services, gathered_feedback
     # Define incoming_msg at the start
    incoming_msg = request.values.get('Body', '').lower()  # Ensure incoming_msg is defined
    try:
        phone_number = WHATSAPP_USER_PHONE_NUMBER
        customer_status = get_customer_status(phone_number)
        
        if not greeted:
            if customer_status == "new":
                message = whatsapp_client.messages.create(
                    body="Hi, this is Tau from Star International. How are you doing today?",
                    from_=whatsapp_number,
                    to=phone_number
                )
            else:
                message = whatsapp_client.messages.create(
                    body="Hi, this is Tau from Star International. How can we assist you today?",
                    from_=whatsapp_number,
                    to=phone_number
                )
            greeted = True
        elif not asked_about_wellbeing:
            sentiment = TextBlob(incoming_msg).sentiment.polarity
            if sentiment < 0:
                message = whatsapp_client.messages.create(
                    body="I'm sorry to hear that you're not feeling well. What's wrong?",
                    from_=whatsapp_number,
                    to=phone_number
                )
                asked_about_wellbeing = True
            else:
                message = whatsapp_client.messages.create(
                    body="Glad to hear you're doing well! By the way, do you have any loads for us to carry?",
                    from_=whatsapp_number,
                    to=phone_number
                )
                asked_about_wellbeing = True
                asked_about_services = True
        elif asked_about_wellbeing and not asked_about_services:
            if "yes" in incoming_msg:
                message = whatsapp_client.messages.create(
                    body="Great! Could you please provide more information about the load?",
                    from_=whatsapp_number,
                    to=phone_number
                )
                asked_about_services = True
                gathered_feedback = True
            elif "no" in incoming_msg:
                message = whatsapp_client.messages.create(
                    body="Thank you for your time. If you need any assistance, feel free to reach out to us anytime. Have a great day!",
                    from_=whatsapp_number,
                    to=phone_number
                )
                interaction_counter = 0  # Reset counter after interaction ends
                greeted = False  # Reset greeting flag
                asked_about_wellbeing = False
                asked_about_services = False
                gathered_feedback = False
            else:
                message = whatsapp_client.messages.create(
                    body="Sorry, I didn't understand that. Can you please tell me if you have any loads for us to carry?",
                    from_=whatsapp_number,
                    to=phone_number
                )
        else:
            message = whatsapp_client.messages.create(
                body="Thank you for the information. Is there anything else we can help you with?",
                from_=whatsapp_number,
                to=phone_number
            )
        
        # Logging WhatsApp message initiation
        app.logger.info(f"WhatsApp message sent successfully with SID: {message.sid}")
        
        return jsonify({"message": "WhatsApp message sent successfully", "message_sid": message.sid})
    except Exception as e:
        # Logging error
        app.logger.error(f"Error sending WhatsApp message: {e}")
        
        return jsonify({"error": str(e)})

@app.route('/whatsapp', methods=['POST'])
def handle_whatsapp():
    global interaction_counter, greeted, asked_about_loads, asked_about_wellbeing

    # Initialize variables if not already defined
    if 'asked_about_loads' not in globals():
        asked_about_loads = False
    if 'asked_about_wellbeing' not in globals():
        asked_about_wellbeing = False
    if 'greeted' not in globals():
        greeted = False

    # Define incoming_msg at the start
    incoming_msg = request.values.get('Body', '').lower()
    phone_number = request.values.get('From', '')
    customer_status = get_customer_status(phone_number)
    response = MessagingResponse()
    message = response.message()

    # Log the incoming WhatsApp message
    app.logger.info(f"Incoming WhatsApp message: {incoming_msg}")

    # Handle initial greeting
    if not greeted:
        if customer_status == "new":
            message.body("Hi, this is Tau from Star International. How are you doing today?")
        else:
            message.body("Hi, this is Tau from Star International. How can we assist you today?")
        greeted = True
        interaction_counter += 1
        return str(response)

    # Sentiment analysis and follow-up
    if not asked_about_wellbeing:
        sentiment = TextBlob(incoming_msg).sentiment.polarity
        if sentiment < 0:
            message.body("I'm sorry to hear that you're not feeling well. What's wrong?")
            asked_about_wellbeing = True
        else:
            message.body("Glad to hear you're doing well! By the way, do you have any loads for us to carry?")
            asked_about_wellbeing = True
        interaction_counter += 1
        return str(response)

    # Handle the response to the loads question
    if asked_about_wellbeing and not asked_about_loads:
        if "yes" in incoming_msg:
            message.body("Great! Could you please provide more information about the load?")
            asked_about_loads = True
        elif "no" in incoming_msg:
            message.body("Thank you for your time. If you need any assistance, feel free to reach out to us anytime. Have a great day!")
            interaction_counter = 0  # Reset counter
            greeted = False  # Reset greeting flag
            asked_about_loads = False
            asked_about_wellbeing = False
        else:
            message.body("Sorry, I didn't understand that. Can you please tell me if you have any loads for us to carry?")
        return str(response)

    # Use the ask function to get the response from the NLP model
    response_text = ask(incoming_msg, df)
    message.body(response_text)
    interaction_counter += 1

    # Check if the maximum number of interactions has been reached
    if interaction_counter >= MAX_INTERACTIONS:
        message.body("Thank you for reaching out to Star International. We are here whenever you need us. Please feel free to reach out anytime. Have a great day!")
        interaction_counter = 0  # Reset counter
        greeted = False  # Reset greeting flag
        asked_about_loads = False
        asked_about_wellbeing = False
    elif not asked_about_loads:
        # Follow-up marketing message
        message.body("By the way, do you have any loads for us to carry?")
    
    return str(response)


if __name__ == '__main__':
    app.run(port=8000, debug=True)
