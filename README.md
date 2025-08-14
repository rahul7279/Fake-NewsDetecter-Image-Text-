# 🤖 AI-Powered Misinformation Detector

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Google AI](https://img.shields.io/badge/Google_AI-Gemini_1.5-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An advanced, multi-modal fact-checking agent designed to combat misinformation by performing deep analysis on news articles, including their text and associated images.

---

## ✨ Key Features

* **Multi-Modal Analysis:** Fact-checks not just the text of a claim, but also analyzes the accompanying image to detect manipulation or out-of-context usage.
* **Smart Input Handling:** Seamlessly works with four types of inputs:
    * 📰 **Webpage URLs:** Scrapes text and images directly from news articles.
    * 🔗 **Direct Image URLs:** Downloads and analyzes images from direct links.
    * 📄 **Local Files:** Analyzes images directly from your computer.
    * 💬 **Direct Text Claims:** Fact-checks a plain text sentence or claim.
* **2-Step AI Architecture:**
    1.  **Vision Agent:** A specialized agent first analyzes the image to provide a detailed, unbiased description.
    2.  **Fact-Checking Agent:** A second, powerful agent takes the user's claim and the image description, uses **Google Search** to find trusted sources, and provides a comprehensive final verdict.

---

## 🛠️ Tech Stack

* **Core Language:** Python
* **AI Model:** Google Gemini 1.5 Pro Flash
* **AI Framework:** Google Agent Development Kit (ADK)
* **Web Scraping:** Requests & BeautifulSoup
* **Image Processing:** Pillow (PIL)

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/rahul7279/Fake-NewsDetecter-Image-Text-.git](https://github.com/rahul7279/Fake-NewsDetecter-Image-Text-.git)
    cd Fake-NewsDetecter-Image-Text-
    ```
2.  **Set up your Environment Variable:**
    This project requires a Google AI API Key. Set it as an environment variable named `GOOGLE_API_KEY`.

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the script:**
    ```bash
    python NewsDetecter.py
    ```
---

## 🔠Text base Demo

![Image](https://github.com/user-attachments/assets/89974ecb-be0f-4cab-9dbd-e28b76dd95da)

## 🎬Image base Demo

![Image](https://github.com/user-attachments/assets/9488b4fe-8c61-4181-a1b6-3409b17a5683)
