import os
import asyncio
import requests
from bs4 import BeautifulSoup
from PIL import Image
import io
import google.generativeai as genai
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai.types import Content, Part
import warnings
from urllib.parse import urlparse

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

# --- API Key Setup ---
api_key = os.environ.get('GOOGLE_API_KEY')
if not api_key:
    raise Exception("GOOGLE_API_KEY environment variable not set!")
genai.configure(api_key=api_key)
os.environ['GOOGLE_API_KEY'] = api_key
print("✅ API Key configured successfully!")

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
        print(f"Webpage content scrape karne mein dikkat aayi: {e}")
    return scraped_text, scraped_image

def download_direct_image(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        print(f"Direct image download karne mein dikkat aayi: {e}")
    return None

def read_image_from_path(path):
    try:
        return Image.open(path)
    except Exception as e:
        print(f"Local image kholne mein dikkat aayi: {e}")
    return None

# --- Agent Definitions ---
image_vision_agent = Agent( name="image_vision_agent", model="gemini-1.5-flash", instruction="You are an expert image analyst. Describe the provided image in detail. Be concise and factual. Do not use any tools." )
fact_checking_agent = Agent(
    name="fact_checking_agent",
    model="gemini-1.5-flash",
    instruction="""You are an expert Fact-Checking AI assistant. You will be provided with a user's claim and a description of an associated image. Your job is to fact-check the claim using the provided context and your search tool.
    Your final response MUST be in this format:
    **Overall Verdict:** [Credible / Misleading / Fake / Unverified]
    **Analysis:** [Your detailed reasoning and justification.]
    **Sources:** [A list of URLs you found during your search.]""",
    tools=[google_search]
)
print("🤖 Dono agents (Vision & Fact-Checker) taiyaar hain!")

# --- Helper Function to Run Agents ---
async def run_agent_query(agent: Agent, query_parts: list, session_id: str, user_id: str):
    print(f"\n🚀 Agent '{agent.name}' ko request bheja ja raha hai...")
    runner = Runner(agent=agent, session_service=session_service, app_name=agent.name)
    final_response = ""
    try:
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=Content(parts=query_parts, role="user")):
            if event.is_final_response():
                final_response = event.content.parts[0].text
    except Exception as e:
        final_response = f"An error occurred: {e}"
    return final_response

# --- Main Function (Updated with Direct Text Handling) ---
session_service = InMemorySessionService()
my_user_id = "adk_adventurer_001"

async def run_news_checker_query():
    user_input = input("Enter a URL, Local Image Path, OR a Text Claim to verify: ").strip()
    if not user_input:
        print("Aapne kuch enter nahi kiya.")
        return

    text_content, image_content = "", None
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
    
    # Naya Logic: Ab yeh direct text, URL, aur file path, teeno ko handle karega
    if user_input.lower().startswith('http'):
        parsed_path = urlparse(user_input).path.lower()
        if any(parsed_path.endswith(ext) for ext in image_extensions):
            print("🔗 Direct Image URL detect hua...")
            image_content = download_direct_image(user_input)
            if image_content: text_content = input("Is image ke saath juda daava (claim) likhein: ")
        else:
            print("📰 Webpage URL detect hua...")
            text_content, image_content = scrape_webpage_content(user_input)
    elif os.path.exists(user_input):
        print("📄 Local file path detect hua...")
        image_content = read_image_from_path(user_input)
        if image_content: text_content = input("Is image ke saath juda daava (claim) likhein: ")
    else:
        # Agar input na to URL hai na hi file path, to use direct text claim maana jayega
        print("💬 Direct text claim detect hua...")
        text_content = user_input
        image_content = None

    if not text_content and not image_content:
        print("⚠️ Koi content nahi mil saka. Jaanch nahi ho sakti.")
        return
        
    if image_content: print("✅ Image loaded successfully.")
    if text_content: print("✅ Text loaded successfully.")

    image_description = "No image was provided."
    # STEP 1: Agar image hai to use analyze karna
    if image_content:
        print("✅ Image loaded. Ab image ka vivaran nikala ja raha hai...")
        if image_content.mode != 'RGB': image_content = image_content.convert('RGB')
        vision_session = await session_service.create_session(app_name=image_vision_agent.name, user_id=my_user_id)
        image_query_parts = [Part(text="Describe this image in detail."), image_content]
        image_description = await run_agent_query(agent=image_vision_agent, query_parts=image_query_parts, session_id=vision_session.id, user_id=my_user_id)
        print("✅ Image ka vivaran mil gaya.")

    # STEP 2: Fact-checking karna
    final_query_for_fact_checker = f"""
    Please fact-check the following user claim based on the provided context.
    **User's Claim:** "{text_content}"
    **Context from Associated Image:** "{image_description}"
    """
    fact_check_session = await session_service.create_session(app_name=fact_checking_agent.name, user_id=my_user_id)
    final_response = await run_agent_query(agent=fact_checking_agent, query_parts=[Part(text=final_query_for_fact_checker)], session_id=fact_check_session.id, user_id=my_user_id)

    print("\n" + "#"*50)
    print("🎉 FINAL FACT-CHECK REPORT 🎉")
    print("#"*50)
    print(final_response)
    print("#"*50 + "\n")

# --- Program Execution ---
if __name__ == "__main__":
    asyncio.run(run_news_checker_query())