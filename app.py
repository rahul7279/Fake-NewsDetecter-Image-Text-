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
from google.adk.sessions import InMemorySessionService, Session
from google.genai.types import Content, Part, Blob
import warnings
from urllib.parse import urlparse

# --- Page ki initial configuration ---
st.set_page_config(
    page_title="AI Fact-Checker", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Ignore Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    body, h1, h2, h3, p {
        font-family: 'Roboto', sans-serif !important;
    }
    .stApp { background-color: #121212; }
    .st-emotion-cache-1y4p8pa { max-width: 80%; } /* Chat container width */
    [data-testid="stChatMessage"] { 
        background-color: #282828; 
        border-radius: 12px; 
        border: 1px solid #404040; 
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    [data-testid="stHeader"] { background-color: transparent; }
    h1 { 
        font-weight: 700; 
        color: #FFFFFF; 
        text-align: center; 
        padding-bottom: 10px; 
        border-bottom: 2px solid #00FF7F;
    }
</style>
""", unsafe_allow_html=True)

# --- API Key Setup ---
try:
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        st.error("Error: GOOGLE_API_KEY environment variable set nahi hai!")
        st.stop()
    genai.configure(api_key=api_key)
    os.environ['GOOGLE_API_KEY'] = api_key
except Exception as e:
    st.error(f"API Key configure karne mein error aaya: {e}")
    st.stop()

# --- Helper Functions ---
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

# --- Agent Definitions ---
image_vision_agent = Agent(name="image_vision_agent", model="gemini-1.5-flash", instruction="You are an expert image analyst. Describe the provided image in detail. Be concise and factual. Do not use any tools.")
fact_checking_agent = Agent(name="fact_checking_agent", model="gemini-1.5-flash", instruction="""You are an expert Fact-Checking AI assistant. You will be provided with a user's claim and a description of an associated image. Your job is to fact-check the claim using the provided context and your search tool.
    Your final response MUST be in this format:
    **Overall Verdict:** [Credible / Misleading / Fake / Unverified]
    **Analysis:** [Your detailed reasoning and justification.]
    **Sources:** [A list of URLs you found during your search.]""", tools=[google_search])

# --- Helper Function to Run Agents ---
session_service = InMemorySessionService()
my_user_id = "streamlit_user_001"
async def run_agent_query(agent: Agent, query_parts: list, session_id: str, user_id: str):
    runner = Runner(agent=agent, session_service=session_service, app_name=agent.name)
    final_response = ""
    try:
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=Content(parts=query_parts, role="user")):
            if event.is_final_response():
                final_response = event.content.parts[0].text
    except Exception as e:
        final_response = f"An error occurred: {e}"
    return final_response

# --- Main Logic Function ---
async def process_fact_checking(text_content, image_content):
    image_description = "No image was provided."
    if image_content:
        if image_content.mode != 'RGB': image_content = image_content.convert('RGB')
        
        img_byte_arr = io.BytesIO()
        image_content.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()
        image_part = Part(inline_data=Blob(data=image_bytes, mime_type='image/jpeg'))
        
        vision_session = await session_service.create_session(app_name=image_vision_agent.name, user_id=my_user_id)
        image_query_parts = [Part(text="Describe this image in detail."), image_part]
        image_description = await run_agent_query(agent=image_vision_agent, query_parts=image_query_parts, session_id=vision_session.id, user_id=my_user_id)
    
    final_query = f"""
    Please fact-check the following user claim based on the provided context.
    **User's Claim:** "{text_content}"
    **Context from Associated Image:** "{image_description}"
    """
    fact_check_session = await session_service.create_session(app_name=fact_checking_agent.name, user_id=my_user_id)
    final_report = await run_agent_query(agent=fact_checking_agent, query_parts=[Part(text=final_query)], session_id=fact_check_session.id, user_id=my_user_id)
    
    return final_report


# --- APP UI ---
st.title("🤖 AI Fact-Checker")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Main aapka AI Fact-Checker hoon. Aap mujhse koi text, URL, ya image check karwa sakte hain."}]

with st.sidebar:
    st.header("Image Upload")
    st.info("Agar aapko image check karani hai, to use yahan upload karein, aur usse juda daava (claim) neeche chat box mein likhein.")
    uploaded_image_file = st.file_uploader("Image chunein...", type=['jpg', 'jpeg', 'png'])
    if uploaded_image_file:
        st.session_state.uploaded_image = read_image_from_file(uploaded_image_file)
        st.image(st.session_state.uploaded_image, caption="Image loaded.", use_column_width=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("URL, text claim, ya image ke baare mein sawaal yahan likhein..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analysis in progress..."):
            text_content, image_content = "", None
            image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
            user_input = prompt
            
            if 'uploaded_image' in st.session_state and st.session_state.uploaded_image:
                image_content = st.session_state.uploaded_image
                text_content = user_input
                st.session_state.uploaded_image = None
            elif user_input.lower().startswith('http'):
                parsed_path = urlparse(user_input).path.lower()
                if any(parsed_path.endswith(ext) for ext in image_extensions):
                    image_content = download_direct_image(user_input)
                    text_content = "User provided a direct image link. The claim is to analyze this image."
                else:
                    text_content, image_content = scrape_webpage_content(user_input)
            else:
                text_content = user_input
            
            # Run the main logic asynchronously
            final_report = asyncio.run(process_fact_checking(text_content, image_content))

            st.markdown(final_report)
            st.session_state.messages.append({"role": "assistant", "content": final_report})