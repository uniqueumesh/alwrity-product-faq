import streamlit as st
import os
import json
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential
import google.generativeai as genai
from bs4 import BeautifulSoup
import datetime

# --- Helper functions ---
def get_serp_results(product_keywords, user_serper_api_key=None):
    try:
        serp_results = perform_serperdev_google_search(product_keywords, user_serper_api_key)
        people_also_ask = [item.get("question") for item in serp_results.get("peopleAlsoAsk", [])]
        return serp_results, people_also_ask
    except Exception as e:
        st.warning(f"SERP research failed: {e}")
        return {"peopleAlsoAsk": [], "relatedQuestions": [], "relatedSearches": []}, []

def perform_serperdev_google_search(query, user_serper_api_key=None):
    serper_api_key = user_serper_api_key or os.getenv('SERPER_API_KEY')
    if not serper_api_key:
        st.error("SERPER_API_KEY is missing. Set it in the .env file or provide it in the sidebar.")
        return {"peopleAlsoAsk": [], "relatedQuestions": [], "relatedSearches": []}
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "gl": "in",
        "hl": "en",
        "num": 10,
        "autocorrect": True,
        "page": 1,
        "type": "search",
        "engine": "google"
    })
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    with st.spinner("Searching Google..."):
        try:
            response = requests.post(url, headers=headers, data=payload, stream=True)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error: {response.status_code}, {response.text}")
                return {"peopleAlsoAsk": [], "relatedQuestions": [], "relatedSearches": []}
        except Exception as e:
            st.error(f"SERPER API request failed: {e}")
            return {"peopleAlsoAsk": [], "relatedQuestions": [], "relatedSearches": []}

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_text_with_exception_handling(prompt, user_gemini_api_key=None):
    try:
        api_key = user_gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            st.error("GEMINI_API_KEY is missing. Please provide it in the sidebar or set it in the environment.")
            return None
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 8192,
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        model = genai.GenerativeModel(model_name="models/gemini-2.0-flash", generation_config=generation_config, safety_settings=safety_settings)
        convo = model.start_chat(history=[])
        convo.send_message(prompt)
        return convo.last.text
    except Exception as e:
        st.exception(f"GEMINI: An unexpected error occurred: {e}")
        return None

def extract_product_details_from_url(product_url):
    """
    Scrape product title, features, description, price, and rating from the product URL (enhanced version).
    Returns a dict with keys: title, features, description, price, rating.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(product_url, headers=headers, timeout=10)
        if response.status_code != 200:
            return {}
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.find("title").get_text(strip=True) if soup.find("title") else ""
        desc_tag = soup.find("meta", attrs={"name": "description"})
        description = desc_tag["content"].strip() if desc_tag and desc_tag.has_attr("content") else ""
        features = []
        for ul in soup.find_all("ul"):
            for li in ul.find_all("li"):
                text = li.get_text(strip=True)
                if text and len(text) > 25 and len(features) < 10:
                    features.append(text)
            if features:
                break
        # Try to extract price (common patterns)
        price = ""
        price_selectors = [
            {'name': 'span', 'attrs': {'id': 'priceblock_ourprice'}},
            {'name': 'span', 'attrs': {'id': 'priceblock_dealprice'}},
            {'name': 'span', 'attrs': {'class': 'a-price-whole'}},
            {'name': 'span', 'attrs': {'class': 'price'}},
            {'name': 'div', 'attrs': {'class': 'product-price'}},
        ]
        for sel in price_selectors:
            tag = soup.find(sel['name'], attrs=sel['attrs'])
            if tag:
                price = tag.get_text(strip=True)
                break
        # Try to extract rating (common patterns)
        rating = ""
        rating_selectors = [
            {'name': 'span', 'attrs': {'class': 'a-icon-alt'}},
            {'name': 'span', 'attrs': {'class': 'reviewCountTextLinkedHistogram'}},
            {'name': 'span', 'attrs': {'class': 'averageStarRating'}},
        ]
        for sel in rating_selectors:
            tag = soup.find(sel['name'], attrs=sel['attrs'])
            if tag:
                rating = tag.get_text(strip=True)
                break
        return {
            "title": title,
            "description": description,
            "features": features,
            "price": price,
            "rating": rating
        }
    except Exception as e:
        return {}

def format_serp_for_prompt(serp_results, people_also_ask):
    serp_section = ""
    if serp_results.get("organic"):
        serp_section += "Top Organic Results:\n"
        for idx, item in enumerate(serp_results["organic"][:5], 1):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            serp_section += f"{idx}. {title}: {snippet}\n"
    if serp_results.get("relatedSearches"):
        serp_section += "\nRelated Searches:\n"
        for s in serp_results["relatedSearches"]:
            serp_section += f"- {s.get('query', '')}\n"
    if people_also_ask:
        serp_section += "\nPeople Also Ask:\n"
        for q in people_also_ask:
            serp_section += f"- {q}\n"
    return serp_section.strip()

def extract_seo_keywords_from_serp(serp_results):
    """
    Extracts top keywords from SERP organic results and related searches.
    Returns a list of keywords/phrases.
    """
    keywords = set()
    # Extract from organic titles and snippets
    if serp_results.get("organic"):
        for item in serp_results["organic"]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            for word in title.split():
                if len(word) > 3:
                    keywords.add(word.lower())
            for word in snippet.split():
                if len(word) > 3:
                    keywords.add(word.lower())
    # Extract from related searches
    if serp_results.get("relatedSearches"):
        for s in serp_results["relatedSearches"]:
            query = s.get("query", "")
            for word in query.split():
                if len(word) > 3:
                    keywords.add(word.lower())
    # Return top 15 unique keywords/phrases
    return list(keywords)[:15]

def generate_product_faqs(product_keywords, ecommerce_platform, user_gemini_api_key, user_serper_api_key, product_url, serp_results, people_also_ask, faq_language, faq_count, faq_tone="Default", faq_length="Default", include_seo_keywords=True, seo_keywords=None):
    product_details = extract_product_details_from_url(product_url) if product_url else {}
    serp_prompt = format_serp_for_prompt(serp_results, people_also_ask)
    details_section = ""
    if product_details:
        details_section += "Product Details from URL:\n"
        if product_details.get("title"):
            details_section += f"Title: {product_details['title']}\n"
        if product_details.get("description"):
            details_section += f"Description: {product_details['description']}\n"
        if product_details.get("features"):
            details_section += "Features:\n"
            for feat in product_details["features"]:
                details_section += f"- {feat}\n"
        if product_details.get("price"):
            details_section += f"Price: {product_details['price']}\n"
        if product_details.get("rating"):
            details_section += f"Rating: {product_details['rating']}\n"
    # Add tone, length, and SEO keywords to the prompt
    tone_section = f"\nTone/Style: {faq_tone}." if faq_tone and faq_tone != "Default" else ""
    length_section = ""
    if faq_length == "Short (20-30 words)":
        length_section = "\nEach answer should be 20-30 words."
    elif faq_length == "Medium (40-50 words)":
        length_section = "\nEach answer should be 40-50 words."
    elif faq_length == "Long (60+ words)":
        length_section = "\nEach answer should be at least 60 words."
    seo_section = ""
    if include_seo_keywords and seo_keywords:
        seo_section = f"\nIncorporate these SEO keywords naturally: {', '.join(seo_keywords)}."
    prompt = (
        f"You are an expert e-commerce content writer. Generate {faq_count} unique, concise FAQs for the product '{product_keywords}' on {ecommerce_platform}. "
        f"Use the following SERP research and product details for inspiration. Write in {faq_language}. Format as a numbered list for easy copy-paste."
        f"{tone_section}{length_section}{seo_section}\n\n"
        f"{details_section}\n"
        f"{serp_prompt}\n"
    )
    return generate_text_with_exception_handling(prompt, user_gemini_api_key)

def faqs_to_jsonld(faqs, product_keywords):
    import re
    qas = []
    for line in faqs.split('\n'):
        match = re.match(r"\d+\.\s*(.+?)\?\s*(.+)", line)
        if match:
            question, answer = match.groups()
            qas.append({"question": question.strip() + "?", "answer": answer.strip()})
    if not qas:
        parts = faqs.split('?')
        for i in range(0, len(parts)-1, 2):
            question = parts[i].strip() + '?'
            answer = parts[i+1].strip()
            qas.append({"question": question, "answer": answer})
    jsonld = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": qa["question"],
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": qa["answer"]
                }
            } for qa in qas
        ]
    }
    return json.dumps(jsonld, indent=2)

# --- Competitive Analysis: Display top competitors from SERP ---
def extract_competitors_from_serp(serp_results):
    competitors = []
    if serp_results.get("organic"):
        for item in serp_results["organic"][:5]:
            competitors.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })
    return competitors

def display_competitor_table(competitors):
    if competitors:
        st.markdown('<h5 style="color:#1976D2;">Top Competitor Products from SERP</h5>', unsafe_allow_html=True)
        import pandas as pd
        df = pd.DataFrame(competitors)
        df = df.rename(columns={"title": "Title", "url": "URL", "snippet": "Snippet"})
        st.dataframe(df, use_container_width=True)

# --- Basic Plagiarism Check: flag FAQs that appear verbatim in SERP ---
def check_faq_uniqueness(faqs, serp_results):
    """Returns a list of (faq_line, is_unique) tuples."""
    serp_text = " ".join([
        item.get("title", "") + " " + item.get("snippet", "")
        for item in serp_results.get("organic", [])
    ]).lower()
    result = []
    for line in faqs.split('\n'):
        line_clean = line.strip().lower()
        is_unique = line_clean not in serp_text
        result.append((line, is_unique))
    return result

# --- Main App ---
def main():
    st.set_page_config(
        page_title="ALwrity - AI Product FAQs Generator",
        layout="wide",
    )
    st.markdown("""
        <style>
        ::-webkit-scrollbar-track { background: #e1ebf9; }
        ::-webkit-scrollbar-thumb { background-color: #90CAF9; border-radius: 10px; border: 3px solid #e1ebf9; }
        ::-webkit-scrollbar-thumb:hover { background: #64B5F6; }
        ::-webkit-scrollbar { width: 16px; }
        div.stButton > button:first-child {
            background: #1565C0;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 10px 2px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1>🤖 ALwrity Product FAQs Generator</h1>', unsafe_allow_html=True)
    st.markdown('<div style="color:#1976D2;font-size:1.2rem;margin-bottom:1.5rem;">Generate product FAQs for Amazon and other e-commerce platforms in seconds.</div>', unsafe_allow_html=True)

    with st.expander("API Configuration 🔑", expanded=False):
        user_gemini_api_key = st.text_input("Gemini API Key", type="password")
        user_serper_api_key = st.text_input("SERPER API Key", type="password")

    st.markdown('<h3>2️⃣ Enter Product Details</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        product_keywords = st.text_input('🔑 Product Name/Keywords', placeholder="e.g., wireless earbuds, air fryer")
        ecommerce_platform = st.selectbox('🛒 E-commerce Platform', ('Amazon', 'Flipkart', 'Walmart', 'Other'))
        product_url = st.text_input('🔗 Product URL (optional)', placeholder="https://amazon.com/...")
        faq_count = st.slider('Number of FAQs', min_value=3, max_value=10, value=5)
    with col2:
        faq_language = st.selectbox('🌐 FAQ Output Language', options=["English", "Spanish", "French", "German", "Other"])
        if faq_language == "Other":
            faq_language = st.text_input("Specify Language", placeholder="e.g., Italian, Chinese")
        faq_tone = st.selectbox('🎨 FAQ Tone/Style', options=["Default", "Professional", "Casual", "Persuasive", "Informative"])
        faq_length = st.selectbox('📏 FAQ Length', options=["Default", "Short (20-30 words)", "Medium (40-50 words)", "Long (60+ words)"])
        include_seo_keywords = st.checkbox('Include SEO Keywords', value=True)

    serp_results, people_also_ask = get_serp_results(product_keywords, user_serper_api_key) if product_keywords else ({}, [])
    product_details = extract_product_details_from_url(product_url) if product_url else {}
    seo_keywords = extract_seo_keywords_from_serp(serp_results) if serp_results else []
    competitors = extract_competitors_from_serp(serp_results)
    if product_keywords and (serp_results or people_also_ask):
        st.markdown('<h4 style="color:#1976D2;">🔎 SERP Research Results</h4>', unsafe_allow_html=True)
        if people_also_ask:
            st.markdown('**People Also Ask:**')
            for idx, q in enumerate(people_also_ask, 1):
                st.markdown(f"{idx}. {q}")
        if seo_keywords:
            st.markdown('**Top SEO Keywords from SERP:**')
            st.markdown(", ".join(seo_keywords))
        if competitors:
            display_competitor_table(competitors)
    if product_details:
        st.markdown('<h4 style="color:#1976D2;">📝 Product Details Extracted from URL</h4>', unsafe_allow_html=True)
        if product_details.get("title"):
            st.markdown(f"**Title:** {product_details['title']}")
        if product_details.get("description"):
            st.markdown(f"**Description:** {product_details['description']}")
        if product_details.get("features"):
            st.markdown("**Features:**")
            for feat in product_details["features"]:
                st.markdown(f"- {feat}")
        if product_details.get("price"):
            st.markdown(f"**Price:** {product_details['price']}")
        if product_details.get("rating"):
            st.markdown(f"**Rating:** {product_details['rating']}")

    # --- Track input state for persistence ---
    input_state = {
        'product_keywords': product_keywords,
        'ecommerce_platform': ecommerce_platform,
        'product_url': product_url,
        'faq_count': faq_count,
        'faq_language': faq_language,
        'faq_tone': faq_tone,
        'faq_length': faq_length,
        'include_seo_keywords': include_seo_keywords,
        'seo_keywords': seo_keywords,
    }
    if 'last_input_state' not in st.session_state:
        st.session_state['last_input_state'] = None
    if 'product_faqs' not in st.session_state:
        st.session_state['product_faqs'] = None
    if 'jsonld' not in st.session_state:
        st.session_state['jsonld'] = None
    if 'faqs_ready' not in st.session_state:
        st.session_state['faqs_ready'] = False

    # Detect input changes to reset results
    if st.session_state['last_input_state'] != input_state:
        st.session_state['product_faqs'] = None
        st.session_state['jsonld'] = None
        st.session_state['faqs_ready'] = False
        st.session_state['last_input_state'] = input_state.copy()

    st.markdown('<h3>3️⃣ Generate Product FAQs</h3>', unsafe_allow_html=True)
    if st.button('✨ Generate Product FAQs'):
        with st.spinner("Generating your product FAQs..."):
            if not product_keywords and not product_url:
                st.error('Please enter product keywords or a product URL!')
            else:
                product_faqs = generate_product_faqs(
                    product_keywords, ecommerce_platform, user_gemini_api_key,
                    user_serper_api_key, product_url, serp_results, people_also_ask, faq_language, faq_count,
                    faq_tone, faq_length, include_seo_keywords, seo_keywords
                )
                if product_faqs:
                    st.session_state['product_faqs'] = product_faqs
                    st.session_state['jsonld'] = faqs_to_jsonld(product_faqs, product_keywords)
                    st.session_state['faqs_ready'] = True

    # --- Display results if available ---
    if st.session_state.get('faqs_ready') and st.session_state.get('product_faqs'):
        st.subheader('**🎉 Your Product FAQs! 🚀**')
        # Plagiarism check
        faq_lines = st.session_state['product_faqs'].split('\n')
        uniqueness = check_faq_uniqueness(st.session_state['product_faqs'], serp_results)
        for line, is_unique in uniqueness:
            if is_unique or not line.strip():
                st.markdown(line)
            else:
                st.markdown(f":red_circle: **Potential duplicate:** {line}")
        st.download_button("Copy All FAQs", st.session_state['product_faqs'], file_name="product_faqs.txt")
        st.download_button("Download FAQ Schema (JSON-LD)", st.session_state['jsonld'], file_name="faq_schema.json", mime="application/json")
        st.code(st.session_state['jsonld'], language="json")
        # --- User Feedback Section ---
        st.markdown('---')
        st.markdown('### 🙏 Was this FAQ helpful?')
        col_yes, col_no = st.columns(2)
        with col_yes:
            thumbs_up = st.button('👍 Yes', key='thumbs_up')
        with col_no:
            thumbs_down = st.button('👎 No', key='thumbs_down')
        feedback_text = st.text_input('Any comments or suggestions?', key='feedback_text')
        if thumbs_up or thumbs_down:
            feedback = {
                'timestamp': datetime.datetime.now().isoformat(),
                'product_keywords': product_keywords,
                'faqs': st.session_state['product_faqs'],
                'helpful': bool(thumbs_up),
                'not_helpful': bool(thumbs_down),
                'comment': feedback_text
            }
            feedback_file = 'feedback.json'
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    all_feedback = json.load(f)
            else:
                all_feedback = []
            all_feedback.append(feedback)
            with open(feedback_file, 'w', encoding='utf-8') as f:
                json.dump(all_feedback, f, indent=2)
            st.success('Thank you for your feedback!')

    with st.expander('❓ Help & Troubleshooting', expanded=False):
        st.markdown('''
        - **Not getting results?** Make sure you entered product keywords or a valid product URL.
        - **API key issues?** Double-check your API keys or leave blank to use the default.
        - **Still stuck?** [See our support & documentation](https://github.com/uniqueumesh/alwrity-faq)
        ''')
    st.markdown('<div class="footer">Made with ❤️ by ALwrity | <a href="https://github.com/uniqueumesh/alwrity-faq" style="color:#1976D2;">Support</a></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
