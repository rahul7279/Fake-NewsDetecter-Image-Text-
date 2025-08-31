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
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import re
import yt_dlp
from pydub import AudioSegment
import whisper

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Fact-Checker",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Ignore Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

# =========================
# Trusted Sources & Helpers
# =========================
TRUSTED_DOMAINS = [
    "site:factcheck.org",
    "site:snopes.com",
    "site:politifact.com",
    "site:altnews.in",
    "site:boomlive.in",
]
DOMAINS_STRING = " OR ".join(TRUSTED_DOMAINS)

def split_transcript_into_claims(transcript: str, max_claims: int = 8):
    """
    YouTube transcript ko chhote factual claims me todta hai.
    Short/noisy lines, stage directions, timestamps ko skip karta hai.
    """
    if not transcript:
        return []
    # Sentence split
    sentences = re.split(r'(?<=[.?!])\s+', transcript)
    # Clean + filter
    claims = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) < 30:
            continue  # bahut chhoti line skip
        # [Music], [Applause], timestamps, etc. skip
        if (s.startswith('[') and s.endswith(']')) or re.match(r'^\(?\d{1,2}:\d{2}\)?$', s):
            continue
        claims.append(s)
        if len(claims) >= max_claims:
            break
    return claims

# --- Custom CSS ---
st.markdown("""
<style>
    /* [Aapka CSS code yahan - koi badlaav nahi] */
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

# --- Helper Functions & Agent Definitions ---
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

def get_youtube_info(url):
    """YouTube URL se metadata aur transcript nikalta hai (reliable way)."""
    try:
        # Extract video ID using regex (works for youtu.be and youtube.com)
        video_id_match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
        if not video_id_match:
            return {"error": "Invalid YouTube URL format."}
        video_id = video_id_match.group(1)

        # Get transcript (en/hi). If not available, keep empty string.
        transcript = ""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
            transcript = " ".join([item['text'] for item in transcript_list])
        except Exception:
            transcript = ""

        # Get video metadata from YouTube oEmbed (no API key needed)
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        title, author = "Unknown Title", "Unknown Channel"
        try:
            response = requests.get(oembed_url, timeout=10)
            if response.status_code == 200:
                metadata = response.json()
                title = metadata.get("title", title)
                author = metadata.get("author_name", author)
        except Exception:
            pass

        return {
            "title": title,
            "author": author,
            "transcript": transcript[:4000] if transcript else ""  # safe cap
        }

    except Exception as e:
        return {"error": f"YouTube info nikalne mein dikkat aayi: {e}"}

image_vision_agent = Agent(
    name="image_vision_agent",
    model="gemini-1.5-flash",
    instruction=(
        "You are an expert image analyst. Describe the provided image in detail. "
        "If you recognize any public figures, state their names. Be concise and factual. Do not use any tools."
    )
)

# ==== UPDATED: fact_checking_agent with trusted sources + per-claim evaluation ====
fact_checking_agent = Agent(
    name="fact_checking_agent",
    model="gemini-1.5-flash",
    instruction=f"""
You are an expert Fact-Checking AI assistant with memory. Always consider conversation history.

WHEN SEARCHING:
- Use the google_search tool and RESTRICT queries to these domains first:
  {DOMAINS_STRING}
- If no relevant results on these, you may widen the search but prefer reputable outlets (major news orgs, .gov, academic).

WHEN INPUT CONTAINS A NUMBERED/BULLETED LIST OF CLAIMS:
- Evaluate EACH claim separately with a short verdict per claim: [True / False / Partly True / Unverified], with 1–2 strongest sources per claim.
- Then give an Overall Verdict for the whole input.

OUTPUT FORMAT (strict):
**Overall Verdict:** [Credible / Misleading / Fake / Unverified]
**Analysis:**
- If multiple claims: list them as 1), 2), ... with mini-verdicts and reasoning.
- If single claim: concise reasoning referencing sources.
**Sources:**
- Bullet list of URLs actually used (put the strongest 3–6). Prefer the trusted domains above when available.
""",
    tools=[google_search]
)

session_service = InMemorySessionService()
my_user_id = "streamlit_user_001"

async def run_agent_query(agent: Agent, query_parts: list, session_id: str):
    runner = Runner(agent=agent, session_service=st.session_state.session_service, app_name=agent.name)
    final_response = ""
    try:
        async for event in runner.run_async(
            user_id=my_user_id,
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
            app_name=image_vision_agent.name,
            user_id=my_user_id
        )
        image_query_parts = [Part(text="Describe this image in detail."), image_part]
        image_description = await run_agent_query(
            agent=image_vision_agent,
            query_parts=image_query_parts,
            session_id=vision_session.id
        )

    # ==== UPDATED: stronger prompt with trusted sources + per-claim instruction ====
    final_query = f"""
You will perform fact-checking.

- If the claim block includes numbered 'Transcript Claims', evaluate EACH claim separately.
- Prefer searching these domains: {DOMAINS_STRING}. If nothing relevant, you may expand but cite the strongest sources only.
- Keep the output strictly in the required format.

**User's Claim or Claims Block:**
{text_content}

**Context from Associated Image (if any):**
{image_description}
"""
    final_report = await run_agent_query(
        agent=fact_checking_agent,
        query_parts=[Part(text=final_query)],
        session_id=fact_check_session_id
    )
    return final_report

def download_audio_from_youtube(url, filename="temp_audio.mp3"):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": filename,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return filename
    except Exception as e:
        return None

def audio_to_transcript(audio_file):
    model = whisper.load_model("base")  # "tiny", "small", "medium" bhi use kar sakte ho
    result = model.transcribe(audio_file, language="en")  # Hindi video ho to language="hi"
    return result["text"]

# --- Main App Logic ---
st.title("🤖 AI Fact-Checker")

if "session_service" not in st.session_state:
    st.session_state.session_service = InMemorySessionService()
    new_session = asyncio.run(
        st.session_state.session_service.create_session(
            app_name=fact_checking_agent.name,
            user_id=my_user_id
        )
    )
    st.session_state.fact_check_session_id = new_session.id
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! Main aapka AI Fact-Checker hoon. Aap mujhse koi text, URL, ya image check karwa sakte hain."
    }]

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
    with st.chat_message("user"):
        st.image(
            st.session_state.uploaded_image,
            caption="Image ready for analysis. Now type your question below.",
            width=200
        )

if prompt := st.chat_input("URL, text claim, ya image ke baare mein sawaal yahan likhein..."):
    image_to_process = st.session_state.get("uploaded_image", None)
    st.session_state.messages.append({"role": "user", "content": prompt, "image": image_to_process})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysis in progress..."):
            text_content, image_content = "", None
            user_input = prompt

            # ==== UPDATED: YouTube branch with transcript→claims + trusted sources ====
            if "youtube.com" in user_input.lower() or "youtu.be" in user_input.lower():
                st.info("YouTube link detect hua, transcript aur metadata nikaal rahe hain...")
                video_info = get_youtube_info(user_input)

                if video_info and "error" not in video_info:
                    transcript = video_info.get("transcript", "")
                    if not transcript:
                        # Agar YouTube transcript nahi mila
                        st.info("Transcript not found, extracting audio for speech-to-text...")
                        audio_file = download_audio_from_youtube(user_input, "temp_audio.mp3")
                        if audio_file:
                            transcript = audio_to_transcript(audio_file)

                    claims = split_transcript_into_claims(transcript, max_claims=8)

                    if claims:
                        # Structured, per-claim fact-checking input
                        text_content = (
                            f"Fact-check the following YouTube video based on transcript claims.\n"
                            f"Title: '{video_info.get('title','Unknown')}'\n"
                            f"Channel: '{video_info.get('author','Unknown')}'\n\n"
                            f"IMPORTANT: Restrict searches to these domains first: {DOMAINS_STRING}\n"
                            f"Evaluate EACH claim separately with a mini-verdict.\n\n"
                            f"Transcript Claims:\n"
                        )
                        for i, c in enumerate(claims, 1):
                            text_content += f"{i}. {c}\n"
                    else:
                        # Fallback: no transcript -> at least use title/channel
                        text_content = (
                            f"Fact-check this YouTube video based on any available context.\n"
                            f"Title: '{video_info.get('title','Unknown')}'\n"
                            f"Channel: '{video_info.get('author','Unknown')}'\n"
                            f"Note: Transcript not available. Use trusted sources search: {DOMAINS_STRING}\n"
                        )

                    image_content = None
                else:
                    text_content = (
                        "YouTube video ki jaankari nahi mil saki. "
                        f"Please cross-check with trusted sources: {DOMAINS_STRING}"
                    )
                    image_content = None

            elif st.session_state.uploaded_image:
                # Image flow (vision agent context + fact-check)
                image_content = st.session_state.uploaded_image
                text_content = user_input
            else:
                # Plain text/URL flow
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
