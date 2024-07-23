import fitz
from io import BytesIO
from PIL import Image
import pytesseract



def extract_text_from_images(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()

        image_xrefs = page.get_images()
        if image_xrefs:
            for img_xref in image_xrefs:
                xref_value = img_xref[0]
                img_dict = doc.extract_image(xref_value)
                img_binary = img_dict["image"]
                try:
                    image_io = BytesIO(img_binary)
                    image = Image.open(image_io)
                    extracted_text = pytesseract.image_to_string(image)
                    text += extracted_text + '\n'
                    image_io.close()
                except Exception as e:
                    print(f"Error processing image: {e}")

    doc.close()
    return text





pdf_path = 'AA3117 instruction de travail sertissage.pdf'
extracted_text = extract_text_from_images(pdf_path)
#print(extracted_text)



#fonction pour diviser le texte extrait en sections basées sur les pages
import re
sections = re.split(r'\bPage \d+ of \d+\b', extracted_text)



# Remove any empty sections and strip whitespace
sections = [section.strip() for section in sections if section.strip()]





#foction pour netoyer le texte extraire
def clean_text(text):
    # Remove special characters, periods, and extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\.', '', text)  # Remove periods
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading and trailing spaces
    return text





for section in sections:
   section = clean_text(section)



from sentence_transformers import SentenceTransformer, util
import torch



#fonction pour avoir la partie du texte la plus adapté a la question
def get_best_matching_context(question, contexts):
    try:
        # Load pre-trained model
        #print("Loading model...")
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        #print("Model loaded successfully.")


        # Generate embeddings for the question
        #print("Generating question embedding...")
        question_embedding = model.encode(question, convert_to_tensor=True)
        #print(f"Question embedding: {question_embedding}")


        # Generate embeddings for the contexts
        #print("Generating context embeddings...")
        context_embeddings = model.encode(contexts, convert_to_tensor=True)
        #print(f"Context embeddings: {context_embeddings}")


        # Compute cosine similarities
        #print("Computing similarities...")
        similarities = util.pytorch_cos_sim(question_embedding, context_embeddings)[0]
        #print(f"Similarities: {similarities}")


        # Find the index of the best matching context
        best_match_idx = torch.argmax(similarities).item()
        best_match_score = similarities[best_match_idx].item()

        return contexts[best_match_idx], best_match_score


    except Exception as e:
        print(f"Error in get_best_matching_context: {e}")
        return None







cleaned_sections = [clean_text(section) for section in sections]
#cleaned_sections







import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

from peft import LoraConfig, PeftModel


from huggingface_hub import login
login(token = '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')





import locale
locale.getpreferredencoding = lambda: "UTF-8"



def generate_response(user_question, cleaned_sections, seuil=0.3):
    best_context, similarity = get_best_matching_context(user_question, cleaned_sections)
    
    
    # Vérifier si la similarité dépasse le seuil
    if similarity >= seuil and best_context is not None:
      
      model_name = "mistralai/Mistral-7B-Instruct-v0.2"

      lora_r = 64
      lora_alpha = 16
      lora_dropout = 0.1
      
      
      use_4bit = True
      bnb_4bit_compute_dtype = "float16"
      bnb_4bit_quant_type = "nf4"
      use_nested_quant = False
      
      
      max_seq_length = None
      packing = False
      device_map = {"": 0}
      
      
      compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
      
      
      bnb_config = BitsAndBytesConfig(
          load_in_4bit=use_4bit,
          bnb_4bit_quant_type=bnb_4bit_quant_type,
          bnb_4bit_compute_dtype=compute_dtype,
          bnb_4bit_use_double_quant=use_nested_quant,
      )


      if compute_dtype == torch.float16 and use_4bit:
          major, _ = torch.cuda.get_device_capability()
          if major >= 8 :
              print("=" * 80)
              print("Your GPU supports bfloat16: accelerate training with bf16=True")
              print("=" * 80)



      model = AutoModelForCausalLM.from_pretrained(
          model_name,
          quantization_config=bnb_config,
          device_map=device_map
      )


      model.config.use_cache = False
      model.config.pretraining_tp = 1


      tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
      tokenizer.pad_token = tokenizer.eos_token
      tokenizer.padding_side = "right"


      peft_config = LoraConfig(
          lora_alpha=lora_alpha,
          lora_dropout=lora_dropout,
          r=lora_r,
          bias="none",
          task_type="CAUSAL_LM",
      )



      device = "cuda" if torch.cuda.is_available() else "cpu"


      messages = [
        {"role": "user", "content": f"Vous êtes un assistant intelligent. Vous recevrez en entrée une question sur le contexte fourni, puis vous y répondrezr. la reponse doit etre en français.  contexte: {best_context}"},
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": user_question}
      ]


      encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")


      model_inputs = encodeds.to(device)


      generated_ids = model.generate(model_inputs, max_new_tokens=500, do_sample=True)
      decoded = tokenizer.batch_decode(generated_ids)
      decoded_text = decoded[0]
      last_inst_index = decoded_text.rfind('[/INST]')


      if last_inst_index != -1:
          response = decoded_text[last_inst_index + len('[/INST]'):].strip()
          # Remove the </s> tag if it exists
          if response.endswith('</s>'):
              response = response[:-5]

          # Calculer le pourcentage d'exactitude
          accuracy_percentage = round(similarity * 100, 2)
          response += f"\n\nExactitude de la réponse: {accuracy_percentage}%"


      return response
    else: 
      return "Je suis désolé, je ne comprends pas bien votre question ou je n'ai pas assez d'informations pour répondre."




import streamlit as st
import os
import shelve
import sentence_transformers
from PIL import Image


# App title
st.title("Chatbot Leoni")

# Logo (if available)
logo_path = os.path.join(os.getcwd(), "Leoni.svg.png")
if os.path.exists(logo_path):
    st.image(logo_path, width=50)

# Welcome message
st.write("**Bonjour, De quoi avez-vous besoin aujourd'hui ?**")



# Icons (using FontAwesome)
user_icon = st.empty()
assistant_icon = st.empty()

user_icon.markdown(f"<i class='fas fa-user-circle' ></i>", unsafe_allow_html=True)
assistant_icon.markdown(f"<i class='fas fa-robot' ></i>", unsafe_allow_html=True)

# Load chat history (shelve)
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

# Save chat history (shelve)
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Create chat container (moved outside conditional for better organization)
chat_container = st.container()
    

# Sidebar 
with st.sidebar:

    # New conversation button
    if st.button("New Conversation"):
        st.session_state.messages = []  # Clear previous conversation history
        st.session_state['conversation_count'] = 0  # Reset conversation count
        chat_container.empty()  # Clear chat display container

# Display chat messages with avatars
def display_chat_messages(messages, container):
    for message in messages:
        avatar = user_icon if message["role"] == "user" else assistant_icon
        with container.chat_message(message["role"]):
            st.markdown(message["content"])




def display_survey():
    """Displays the Marcom Robot survey"""
    st.markdown("""
    <style>
    .survey-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        text-align: center;
        font-family: Arial;
        margin-bottom: 20px;
    }
    .survey-container p {
        font-size: 16px;
        margin-bottom: 10px;
    }
    .survey-options {
        display: flex;
        justify-content: center;
        gap: 10px;
    }
    .survey-option {
        text-decoration: none;
        text-align: center;
        color: inherit;
    }
    .survey-option img {
        border: none;
        margin: 0 auto;
    }
    .survey-option div {
        margin-top: 5px;
    }
    .survey-footer {
        margin-top: 8px;
        font-size: 12px;
        line-height: 18px;
    }
    </style>

    <div class="survey-container">
        <p>Comment vous trouvez cette réponse ?</p>
        <div class="survey-options">
            <a href="https://survey.marcomrobot.com/one-click-survey/collect-response?hash=fd8dacd3b09c0512383b9ed90ded050b124a138dba665a53f267a1ec6ba676137a3d9c656c039b2c3418627546525463&amp;score=1" class="survey-option">
                <img src="https://survey.marcomrobot.com/api/icon/face-angry?color=%23B71C1C" alt="Bad" title="Bad" width="48">
                <div style="color: #B71C1C;">Bad</div>
            </a>
            <a href="https://survey.marcomrobot.com/one-click-survey/collect-response?hash=fd8dacd3b09c0512383b9ed90ded050b124a138dba665a53f267a1ec6ba676137a3d9c656c039b2c3418627546525463&amp;score=2" class="survey-option">
                <img src="https://survey.marcomrobot.com/api/icon/face-frown?color=%23FF8A80" alt="Poor" title="Poor" width="48">
                <div style="color: #FF8A80;">Poor</div>
            </a>
            <a href="https://survey.marcomrobot.com/one-click-survey/collect-response?hash=fd8dacd3b09c0512383b9ed90ded050b124a138dba665a53f267a1ec6ba676137a3d9c656c039b2c3418627546525463&amp;score=3" class="survey-option">
                <img src="https://survey.marcomrobot.com/api/icon/face-meh?color=%23FDD835" alt="Average" title="Average" width="48">
                <div style="color: #FDD835;">Average</div>
            </a>
            <a href="https://survey.marcomrobot.com/one-click-survey/collect-response?hash=fd8dacd3b09c0512383b9ed90ded050b124a138dba665a53f267a1ec6ba676137a3d9c656c039b2c3418627546525463&amp;score=4" class="survey-option">
                <img src="https://survey.marcomrobot.com/api/icon/face-face?color=%2381C784" alt="Good" title="Good" width="48">
                <div style="color: #81C784;">Good</div>
            </a>
            <a href="https://survey.marcomrobot.com/one-click-survey/collect-response?hash=fd8dacd3b09c0512383b9ed90ded050b124a138dba665a53f267a1ec6ba676137a3d9c656c039b2c3418627546525463&amp;score=5" class="survey-option">
                <img src="https://survey.marcomrobot.com/api/icon/face-stars?color=%234CAF50" alt="Excellent" title="Excellent" width="48">
                <div style="color: #4CAF50;">Excellent</div>
            </a>
        </div>
        <div class="survey-footer">
            Votre avis est précieux et ne prendra que 5 secondes. <br>
            Cliquez ou appuyez sur l'évaluation qui reflète le mieux votre expérience.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Chat logic
if user_question := st.chat_input("Saisissez une requête ici"):
    # Generate conversation ID
    st.session_state['conversation_count'] = st.session_state.get('conversation_count', 0) + 1


    # Generate response (assuming response generation functions exist)
    response = generate_response(user_question, cleaned_sections)
    

    

    # Save chat history
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
    save_chat_history(st.session_state.messages)

    display_survey()

    # Display updated chat history (including new messages)
    display_chat_messages(st.session_state.messages, chat_container)
