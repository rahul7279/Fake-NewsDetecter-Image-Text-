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
import re
from youtube_transcript_api import YouTubeTranscriptApi

# --------------------------
# Page & Global Config
# --------------------------
st.set_page_config(
    page_title="AI Fact-Checker",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

# --------------------------
# State
# --------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# --------------------------
# Theme (hard-contrast variant)
# --------------------------
def apply_theme_strict(dark: bool = True):
    # Backgrounds are soft; text and borders are pure black (light) or pure white (dark)
    if dark:
        bg_page = "#000000"
        surface = "#141826"
        text_color = "#ffffff"
        border_color = "#ffffff"
        toggle_outline = "#ffffff"
        accent = "#00e5cf"
        shadow = "0 2px 10px rgba(0,0,0,0.35)"
    else:
        bg_page = "#f0f2f6"
        surface = "#ffffff"
        text_color = "#000000"
        border_color = "#000000"
        toggle_outline = "#000000"
        accent = "#007f7a"
        shadow = "0 2px 8px rgba(0,0,0,0.08)"

    css = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
      .stApp {{ background: {bg_page}; }}
      .main .block-container {{ padding-top: 1.25rem; max-width: 760px; margin: 0 auto; }}

      /* ENFORCE TEXT COLOR GLOBALLY */
      html, body, .stApp, .main, .block-container, .stMarkdown, .stChatMessage,
      [data-testid="stFileUploader"], .stAlert, .stButton, .stTextInput, .stSelectbox,
      .stTextArea, .stSlider, .stRadio, label, p, h1, h2, h3, h4, h5, h6, span, div {{
        color: {text_color} !important;
        font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica Neue, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      }}

      /* SURFACES */
      .themed-surface, .stChatMessage, .stAlert, [data-testid="stFileUploader"] section {{
        background: {surface} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px;
        box-shadow: {shadow};
      }}
      .stChatMessage {{ margin-bottom: 1rem; padding: 1rem 1.25rem; }}

      /* EXPANDER HEADER */
      details > summary {{
        background: {surface} !important;
        border: 1px solid {border_color} !important;
        color: {text_color} !important;
        border-radius: 10px;
        padding: .6rem .9rem;
        list-style: none;
      }}
      details > summary::-webkit-details-marker {{ display:none; }}

      /* FILE UPLOADER BODY */
      [data-testid="stFileUploader"] section {{ padding: 1rem !important; }}

      /* CHAT INPUT SHELL */
      div[data-testid="stChatInput"] > div:nth-child(2) {{
        background: {surface} !important;
        border: 1px solid {border_color} !important;
        border-radius: 10px !important;
        padding: .5rem .75rem;
        box-shadow: {shadow};
      }}

      /* INPUTS / WIDGETS BORDERS */
      input, textarea, select, [role="spinbutton"], [data-baseweb="input"], [data-baseweb="select"] {{
        color: {text_color} !important;
        border-color: {border_color} !important;
      }}

      /* BUTTONS (generic) */
      .stButton > button {{
        background: {accent} !important;
        color: {text_color} !important;
        border: 2px solid {border_color} !important;
        font-weight: 700;
        border-radius: 8px;
      }}

      /* THEME TOGGLE: ensure clear outline and knob border in both modes */
      div[data-testid="stWidgetLabel"] + div [role="switch"] {{
        outline: 2px solid {toggle_outline} !important;  /* switch container */
        border-radius: 999px !important;
      }}
      div[data-baseweb="switch"] > div {{
        border: 2px solid {toggle_outline} !important;   /* knob track outline */
        border-radius: 999px !important;
      }}

      /* BROWSE FILES BUTTON inside uploader (force outline) */
      [data-testid="stFileUploader"] button,
      [data-testid="stFileUploader"] [data-baseweb="button"] {{
        border: 2px solid {border_color} !important;
        color: {text_color} !important;
        background: transparent !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
      }}

      .soft-divider {{ height: 1px; background: {border_color}; margin: 14px 0 10px 0; opacity: 1; }}

      /* Links and headings */
      a, a:visited {{ color: {text_color} !important; text-decoration: underline; }}
      h1, h2, h3, h4, h5, h6 {{ color: {text_color} !important; }}

#      /* Hide Streamlit default footer/menu if present */
#      [data-testid="stToolbar"] {{ visibility: hidden !important; }}
#      footer {{ visibility: hidden !important; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --------------------------
# API Key Setup
# --------------------------
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

# --------------------------
# Helpers
# --------------------------
def scrape_webpage_content(url):
    scraped_text, scraped_image = "", None
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        scraped_text = "\n".join([p.get_text() for p in paragraphs]) or ""
        if len(scraped_text) < 100:
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

def get_youtube_info(url):
    try:
        video_id_match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
        if not video_id_match:
            return {"error": "Invalid YouTube URL format."}
        video_id = video_id_match.group(1)

        video_info = {"transcript": "(Transcript not available for this video)"}

        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        try:
            response = requests.get(oembed_url, timeout=10)
            if response.status_code == 200:
                metadata = response.json()
                video_info["title"] = metadata.get("title", "Unknown Title")
                video_info["author"] = metadata.get("author_name", "Unknown Channel")
        except Exception:
            video_info["title"] = "Unknown Title"
            video_info["author"] = "Unknown Channel"

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
            transcript = " ".join([item['text'] for item in transcript_list])
            video_info["transcript"] = transcript[:4000]
        except Exception:
            st.info("Is video ke liye transcript available nahi hai. Sirf metadata se analysis kiya jayega.")

        return video_info
    except Exception as e:
        return {"error": f"YouTube info nikalne mein dikkat aayi: {e}"}

# --------------------------
# Agents
# --------------------------
image_vision_agent = Agent(
    name="image_vision_agent",
    model="gemini-1.5-flash",
    instruction=(
        "You are an expert image analyst. Describe the provided image in detail. "
        "If you recognize any public figures, state their names. Be concise and factual. Do not use any tools."
    )
)

# --- Agent Definitions (FINAL UPGRADE) ---
fact_checking_agent = Agent(
    name="fact_checking_agent",
    model="gemini-1.5-flash",
    instruction=f"""
You are a sophisticated, multi-faceted AI analysis agent with memory. Always consider conversation history. Your goal is not just to verify facts, but to provide deep, nuanced understanding.

**Crucial Rule: Language Protocol**
You MUST detect the language of the user's query. Your entire final response must be in the same language(Always consider Hinglish if user gives input in Hinglish).
- If the query is in English, respond in English.
- **If the query is in Hinglish, you MUST respond in natural, conversational Hinglish. Do not use pure, formal Hindi. Mix English words naturally.**
  - **Good Hinglish Example:** "Analysis ke mutabik, yeh claim misleading hai. Is image mein Nitish Kumar nahi hain."

**Core Task: Deep Analysis**
Use the `Google Search` tool to investigate the user's claim. Your analysis MUST include these three sections:

1.  **Fact-Check Analysis:**
    - Provide a primary verdict: [Credible / Misleading / Fake / Unverified] or [Bharosemand / Bhramak / Jhooth / Asatyaapit].
    - Give detailed reasoning for your verdict, referencing the strongest sources you found.

2.  **Propaganda & Bias Analysis:**
    - Scrutinize the language of the source.
    - Explicitly identify any propaganda techniques used (e.g., Loaded Language, Ad Hominem, Fear Appeals).
    - Provide examples from the text if possible. If no bias is found, state that the language appears neutral.

3.  **Broader Context & Counter-Arguments:**
    - Provide brief historical or political context for the claim.
    - Use your search tool to find the most common counter-arguments or opposing viewpoints.
    - Present these counter-arguments neutrally.

**Output Format (Strict):**
**Overall Verdict:** [Your primary verdict]
\n---\n
**Analysis:** [Your fact-checking reasoning.]
\n---\n
**Propaganda & Bias Scan:** [Your analysis of manipulative language.]
\n---\n
**Broader Context:** [Your summary of context and counter-arguments.]
\n---\n
**Sources:**
- Bullet list of the strongest 3-6 URLs you actually used.
""",
    tools=[google_search]
)

# --------------------------
# Async runner
# --------------------------
async def run_agent_query(agent: Agent, query_parts: list, session_id: str):
    runner = Runner(agent=agent, session_service=st.session_state.session_service, app_name=agent.name)
    final_response = ""
    try:
        async for event in runner.run_async(
            user_id="streamlit_user_001",
            session_id=session_id,
            new_message=Content(parts=query_parts, role="user")
        ):
            if event.is_final_response():
                final_response = event.content.parts[0].text
    except Exception as e:
        final_response = f"An error occurred: {e}"
    return final_response

async def get_fact_check_response(text_content, image_content, fact_check_session_id):
    image_description = "No image was provided."
    if image_content:
        if image_content.mode != 'RGB':
            image_content = image_content.convert('RGB')
        img_byte_arr = io.BytesIO()
        image_content.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()
        image_part = Part(inline_data=Blob(data=image_bytes, mime_type='image/jpeg'))

        vision_session = await st.session_state.session_service.create_session(
            app_name=image_vision_agent.name, user_id="streamlit_user_001"
        )
        image_query_parts = [Part(text="Describe this image in detail."), image_part]
        image_description = await run_agent_query(
            agent=image_vision_agent,
            query_parts=image_query_parts,
            session_id=vision_session.id
        )

    final_query = f"""
    **User's Claim/Query:** "{text_content}"
    **Associated Context (if any):** "{image_description}"
    """
    final_report = await run_agent_query(
        agent=fact_checking_agent,
        query_parts=[Part(text=final_query)],
        session_id=fact_check_session_id
    )
    return final_report

# --------------------------
# Session bootstrap
# --------------------------
def init_session():
    if "session_service" not in st.session_state:
        st.session_state.session_service = InMemorySessionService()
    if "fact_check_session_id" not in st.session_state:
        new_session = asyncio.run(
            st.session_state.session_service.create_session(
                app_name=fact_checking_agent.name,
                user_id="streamlit_user_001"
            )
        )
        st.session_state.fact_check_session_id = new_session.id
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! Main aapka AI Fact-Checker hoon. Aap mujhse koi text, URL, ya image check karwa sakte hain."
        }]
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None

# --------------------------
# UI
# --------------------------
def main():
    # Header with visible toggle
    col1, col2 = st.columns([7, 1])
    with col1:
        st.title("🤖 AI Fact-Checker")
        # Removed the extra caption line for a cleaner look
    with col2:
        st.session_state.dark_mode = st.toggle("Dark", value=st.session_state.dark_mode, key="theme_toggle")

    # Apply strict theme
    apply_theme_strict(st.session_state.dark_mode)

    init_session()

    # Removed the friendly callout line for a cleaner homepage

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "image" in message and message["image"] is not None:
                st.image(message["image"], caption="Image being analyzed.", use_column_width=True)

    # Upload section
    with st.expander("Upload an image or paste a URL", expanded=True):
        uploaded_image_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
        # Removed the footer-like caption under uploader as part of cleanup
        # st.caption("Or paste a direct image URL below.")

    if uploaded_image_file:
        st.session_state.uploaded_image = read_image_from_file(uploaded_image_file)
        with st.chat_message("assistant"):
            st.image(
                st.session_state.uploaded_image,
                caption="Image ready for analysis. Now type your question below.",
                use_column_width=True
            )
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Image received. Ask your question or enter a claim for analysis.",
            "image": st.session_state.uploaded_image
        })

    # Chat input
    prompt = st.chat_input("URL, text claim, ya image ke baare mein sawaal yahan likhein...")

    if prompt:
        image_to_process = st.session_state.get("uploaded_image", None)
        st.session_state.messages.append({"role": "user", "content": prompt, "image": image_to_process})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analysis in progress..."):
                text_content, image_content = "", None
                user_input = prompt

                if "youtube.com" in user_input.lower() or "youtu.be" in user_input.lower():
                    st.info("YouTube link detect hua, video ki jaankari nikali ja rahi hai...")
                    video_info = get_youtube_info(user_input)
                    if video_info and "error" not in video_info:
                        text_content = (
                            f"Fact-check this YouTube video.\n"
                            f"Title: '{video_info.get('title','Unknown')}'\n"
                            f"Channel: '{video_info.get('author','Unknown')}'\n\n"
                            f"Transcript Snippet:\n{video_info.get('transcript', '')}"
                        )
                    else:
                        text_content = "YouTube video ki jaankari nahi mil saki."
                    image_content = None
                elif st.session_state.uploaded_image:
                    image_content = st.session_state.uploaded_image
                    text_content = user_input
                else:
                    text_content = user_input

                final_report = asyncio.run(
                    get_fact_check_response(
                        text_content,
                        image_content,
                        st.session_state.fact_check_session_id
                    )
                )
                st.markdown(final_report)
                st.session_state.messages.append({"role": "assistant", "content": final_report})

                if "uploaded_image" in st.session_state:
                    st.session_state.uploaded_image = None

    # Removed the custom footer line for a cleaner look

if __name__ == "__main__":
    main()
