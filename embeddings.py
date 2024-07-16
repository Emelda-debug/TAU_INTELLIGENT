#script to generate embedding file 
import openai
from openai import OpenAI
import pandas as pd
import os

# Set your OpenAI API key
client = OpenAI(
api_key = 'sk-proj-8eXnXyT8ceNk1NCCLCPiT3BlbkFJI6kL5KEpDvMi3eZD9cjn'
)

# Your text data
texts = [
    "Vision: most competitive logistics worldwide- core values are ",
    "passion-means each staff member gives of their best to afford the customer an unforgettable business experience.",
    "Innovation- means staff are encouraged and expected to come up with new business ideas aimed at differentiating our product range for the benefit of our customers.",
    "Teamwork-means contributing jointly towards the accomplishment of our destiny and business vision.",
    "responsible corporate citizen-means the business will be conducted in accordance with the laws of the land with a bias towards uplifting the community within which we operate",
    "inspiration-means encouraging our staff to become business leaders who take advantage of their full potential to grow profits.",
    "Integrity-means our total business operations is anchored on uprightness and strict corporate governance.",
    "Request for quote: it's that simple, fill in the following details ",
    "select mode of transport between air freight, road and rail freight, ocean freight",
    "Our key personnel: the directors and senior management have a wealth of diverse experience spanning from the",
    "following basic transport solutions. international freight forwarding (sea and air), rail transport and road transport",
    "In addition, they also possess strengths in third-party logistics value-added solutions including among others",
    "customs clearing, warehousing, packaging, bar coding",
    "This wealth of experience was acquired in the course of employment within public and leading multinational",
    "and long-established freight forwarding organisations in various capacities including senior managerial positions.",
    "the directors and senior management have a wealth of diverse experience spanning from the following basic transport",
    "solutions. international freight forwarding (sea and air), rail transport and road transport.",
    "History: 2001- we started with a small service",
    "2004- we did consolidation & handling",
    "2012- we expanded to trucking & warehousing",
    "2016- we broadened to freight insurance",
    "This is our team",
    "1. Molly Chinyama - Managing Director",
    "2. Tendai Gwenzi - General Manager ",
    "3. Takunda Mandondo - Human Resources Manager",
    "4. Shamiso Chivizhe - AI Consultant and Project Manager ",
    "5. Emelda Mada - AI Junior Engineer ",
    "6. Tanyaradzwa Chikosha - Sales and Marketing Officer",
    "7. Tapiwa Chitura - Sales and Marketing Officer",
    "8. Denroy Tavada - Operations and Logistics Supervisor", 
    "9. Elwin Hove - Operations and Logistics Officer",
    "10. Eric Roberts - Student on Attachment in Operations and Logistics",
    "11. Kimberly Chinyama - Operations Clerk",
    "12. Elton Chigumbura - Operations Controller",
    "13. Ethan Siziba - Student on Attachment in Operations and Logistics",
    "14. Aaron Kwangwa - Accountant",
    "15. Fortune Chirima - Accounts Clerk",
    "16. Auxillia Bhasera - Student on Attachment in Accounts",
    "17. Dickson Nyakudya - Workshop Charge Hand",
    "18. Lovedale - Auto Mechanic",
    "19. Kudakwashe Pindayi - Student on Attachment in Mechanic",
    "20. Broosh Herriman - Messenger",
    "21. Gabriel Murerwa - International Driver",
    "22. Arnold Kariwo - International Driver",
    "23. Portifar Chihombori - International Driver",
    "24. Munyaradzi Masango - International Driver",
    "25. Nyasha Million - International Driver",
    "26. Crispen Ruziwa - International Driver",
    "27. Motion Kasvaurere - General Hand ",
    "28. Arnold Mukaro - General Hand ",
    "29. Sithembile Mukwati - General Hand ",
    "30. Tendai Delight Malunga - General Hand",
    "Our Services: we have a variety of road freight options to choose from that enables us to ensure that your cargo is delivered to the designated destination.",
    "consolidation services: in the event that your cargo is either less than a truck load (ltl) or less than a container load (lcl) we offer customized solutions for all freight.",
    "air & sea freight forwarding: we are linked to a vast network of supply chain partners that enables us to provide air and sea freight solutions.",
    "customs clearing: for you to experience hassle-free clearing in Zimbabwe leave star international's specialized staff to handle your customs clearance.",
    "customs consultancy: the company is available to give expert advice and assist in resolving any sticking issues that you might have with Zimbabwe revenue authority (Zimra).",
    "We have: One brand office, we have made two hundred and thirty clients happy",
    "Eight hundred and ninety projects completed",
    "Why choose us: we have over 10 years of experience doing above-board logistics.",
    "if you are in the market to buy logistics services you have come to the right place. star international is strategically positioned to provide you with a solution to meet your unique business needs.",
    "our services know no bounds. it transcends geographic and any other perceivable boundaries. indeed you have made the right decision to be part of the star experience.",
    "Integrity: our dealings are always above board. we take a deliberate stance to preserve your image as a law-abiding corporate citizen by abiding by the laws of countries",
    "we operate in and by carrying out our business in a manner that does not prejudice the state or other stakeholders.",
    "customer focus: we listen and make tailor-made solutions at competitive rates.",
    "Our background: pelcravia enterprise t/a star international",
    "we are a logistics services company incorporated in hong kong, South Africa, Mozambique, and Zimbabwe. our ",
    "core business is the provision of logistics solutions throughout the world either directly or through strategic alliances.",
    "if you are in the market to buy logistics services you have come to the right place. star international is strategically",
    "positioned to provide you with a solution to meet your unique business needs.",
    "If you have a question about the supply chain visit or contact us at:",
    "96 willowvale, willowvale, harare.",
    "operations@starinternational.co.zw",
    "+263 8644 086",
    "How do you get a rate? Rate depends on the tonnage and route. ",
    " Please contact us on the given phone number to get an exact rate based on your requirements"
]

# Function to generate embeddings using the new API
def generate_embeddings(texts):
    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            input=[text],
            model='text-embedding-ada-002'
        )
        embedding = response.data[0].embedding
        embeddings.append(embedding)

    return embeddings

# Generate embeddings for the texts
embeddings = generate_embeddings(texts)

# Get Documents folder path
documents_folder = os.path.expanduser(r'C:\Users\user\Documents\TAU_INTELLIGENT') 

# Define save path with filename
save_path = os.path.join(documents_folder, "babe.csv")

# Create DataFrame with texts and their embeddings
df = pd.DataFrame({'text': texts, 'embedding': embeddings})

# Save to CSV with specified path
df.to_csv(save_path, index=False)
