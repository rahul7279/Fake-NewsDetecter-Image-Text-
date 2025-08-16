import streamlit as st
import asyncio
import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
import io
import google.generativeai as genai
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part, Blob
import warnings
from urllib.parse import urlparse

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Fact-Checker", 
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Ignore Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    
    html, body, [class*="st-"], [class*="css-"] {
        font-family: 'Roboto', sans-serif !important;
    }
    .stApp { background-color: #121212; }
    [data-testid="stChatMessage"] { 
        background-color: #282828; 
        border-radius: 12px; 
        border: 1px solid #404040; 
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        width: 95%;
        margin-left: auto;
        margin-right: auto;
    }
    [data-testid="stHeader"] { background-color: transparent; }
    h1 { 
        font-weight: 700; 
        color: #FFFFFF; 
        text-align: center; 
        padding-bottom: 10px; 
    }
    [data-testid="stFileUploader"] { padding: 0; margin-top: 1rem; }
    [data-testid="stFileUploader"] section { padding: 0.5rem; background-color: #282828; }
    [data-testid="stFileUploader"] label { display: none; }
</style>
""", unsafe_allow_html=True)

# --- API Key Setup ---
try:
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        st.error("Error: GOOGLE_API_KEY environment variable not set!")
        st.stop()
    genai.configure(api_key=api_key)
    os.environ['GOOGLE_API_KEY'] = api_key
except Exception as e:
    st.error(f"API Key configure karne mein error aaya: {e}")
    st.stop()

# --- Helper Functions & Agent Definitions ---
# [Saare Helper Functions (scrape, download, read) yahan - koi badlaav nahi]
def scrape_webpage_content(url):
    scraped_text, scraped_image = "", None
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        scraped_text = "\n".join([p.get_text() for p in paragraphs])
        if not scraped_text or len(scraped_text) < 100:
            scraped_text = soup.body.get_text(separator='\n', strip=True)
        image_tag = soup.find('meta', property='og:image')
        if image_tag and image_tag.get('content'):
            image_url = image_tag.get('content')
            image_response = requests.get(image_url, timeout=15)
            scraped_image = Image.open(io.BytesIO(image_response.content))
    except Exception as e:
        st.warning(f"Webpage content scrape karne mein dikkat aayi: {e}")
    return scraped_text, scraped_image

def download_direct_image(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        st.warning(f"Direct image download karne mein dikkat aayi: {e}")
    return None

def read_image_from_file(file):
    try:
        return Image.open(file)
    except Exception as e:
        st.warning(f"Local image kholne mein dikkat aayi: {e}")
    return None

image_vision_agent = image_vision_agent = Agent(
    name="image_vision_agent", 
    model="gemini-1.5-flash", 
    instruction="You are an expert image analyst. Describe the provided image in detail. **If you recognize any public figures or famous people, you must state their names.** Be concise and factual. Do not use any tools."
)
fact_checking_agent = Agent(
    name="fact_checking_agent", 
    model="gemini-1.5-flash", 
    instruction="""You are an expert Fact-Checking AI assistant with memory. You MUST consider the previous conversation history to understand the context of the user's current query.

    **Special Instruction for Person Identification:**
    If the user asks you to identify or compare a named person with an image described in the context, your process is:
    1.  Use your `Google Search` tool to search for **images and physical descriptions** of the named person.
    2.  Analyze the text results from your search to understand the person's appearance, typical attire, and any other identifying features.
    3.  Compare this information with the image description provided in the context.
    4.  Clearly state in your analysis that your conclusion is based on comparing descriptions and information found online, as you cannot perform a direct visual comparison.

    Your final response MUST be in this format:
    **Overall Verdict:** [Credible / Misleading / Fake / Unverified]
    **Analysis:** [Your detailed reasoning and justification.]
    **Sources:** [A list of URLs you found during your search.]""", 
    tools=[google_search]
)

session_service = InMemorySessionService()
my_user_id = "streamlit_user_001"
async def run_agent_query(agent: Agent, query_parts: list, session_id: str):
    runner = Runner(agent=agent, session_service=st.session_state.session_service, app_name=agent.name)
    final_response = ""
    try:
        async for event in runner.run_async(user_id=my_user_id, session_id=session_id, new_message=Content(parts=query_parts, role="user")):
            if event.is_final_response():
                final_response = event.content.parts[0].text
    except Exception as e:
        final_response = f"An error occurred: {e}"
    return final_response

async def get_fact_check_response(full_chat_history, fact_check_session_id):
    # Find the latest image and text from the history
    latest_user_message = next((msg for msg in reversed(full_chat_history) if msg["role"] == "user"), None)
    
    text_content = latest_user_message["content"]
    image_content = latest_user_message.get("image", None)

    image_description = "No image was provided in the latest turn."
    if image_content:
        if image_content.mode != 'RGB': image_content = image_content.convert('RGB')
        img_byte_arr = io.BytesIO()
        image_content.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()
        image_part = Part(inline_data=Blob(data=image_bytes, mime_type='image/jpeg'))
        
        vision_session = await st.session_state.session_service.create_session(app_name=image_vision_agent.name, user_id=my_user_id)
        image_query_parts = [Part(text="Describe this image in detail."), image_part]
        image_description = await run_agent_query(agent=image_vision_agent, query_parts=image_query_parts, session_id=vision_session.id)
    
    final_query = f"""
    Please fact-check the following user claim based on the provided context.
    **User's Claim:** "{text_content}"
    **Context from Associated Image (if any):** "{image_description}"
    """
    final_report = await run_agent_query(agent=fact_checking_agent, query_parts=[Part(text=final_query)], session_id=fact_check_session_id)
    return final_report

# --- Main App Logic ---
st.title("🤖 AI Fact-Checker")

if "session_service" not in st.session_state:
    st.session_state.session_service = InMemorySessionService()
    new_session = asyncio.run(st.session_state.session_service.create_session(app_name=fact_checking_agent.name, user_id=my_user_id))
    st.session_state.fact_check_session_id = new_session.id
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Main aapka AI Fact-Checker hoon. Aap mujhse koi text, URL, ya image check karwa sakte hain."}]

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message and message["image"] is not None:
             st.image(message["image"], caption="Image being analyzed.", width=200)

uploaded_image_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
if uploaded_image_file:
    st.session_state.uploaded_image = read_image_from_file(uploaded_image_file)
    # Don't add to history yet, just show it
    with st.chat_message("user"):
        st.image(st.session_state.uploaded_image, caption="Image ready for analysis. Now type your question below.", width=200)

if prompt := st.chat_input("URL, text claim, ya image ke baare mein sawaal yahan likhein..."):
    # When user types, attach the uploaded image (if any) to this message
    image_to_process = st.session_state.get("uploaded_image", None)
    st.session_state.messages.append({"role": "user", "content": prompt, "image": image_to_process})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysis in progress..."):
            final_report = asyncio.run(get_fact_check_response(st.session_state.messages, st.session_state.fact_check_session_id))
            st.markdown(final_report)
            st.session_state.messages.append({"role": "assistant", "content": final_report})
            
            # Clear the temporary image holder after it's been attached to a message
            if "uploaded_image" in st.session_state:
                st.session_state.uploaded_image = None