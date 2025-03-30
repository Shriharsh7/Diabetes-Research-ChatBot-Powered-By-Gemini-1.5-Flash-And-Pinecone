import gradio as gr
import pinecone
import requests
import PyPDF2
from transformers import AutoTokenizer, AutoModel
import torch
import re
import google.generativeai as genai
import os
import time
from datetime import datetime, timedelta
from google.api_core import exceptions

# Constants
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Set in HF Spaces Secrets
PINECONE_INDEX_NAME = "diabetes-bot"
PINECONE_NAMESPACE = "general"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set in HF Spaces Secrets
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

# Free tier limits
FREE_TIER_RPD_LIMIT = 1500  # Requests per day
FREE_TIER_RPM_LIMIT = 15    # Requests per minute
FREE_TIER_TPM_LIMIT = 1000000  # Tokens per minute
WARNING_THRESHOLD = 0.9  # Stop at 90% of the limit to be safe

# Usage tracking
usage_file = "usage.txt"

def load_usage():
    if not os.path.exists(usage_file):
        return {"requests": [], "tokens": []}
    with open(usage_file, "r") as f:
        data = f.read().strip()
        if not data:
            return {"requests": [], "tokens": []}
        requests, tokens = data.split("|")
        return {
            "requests": [float(t) for t in requests.split(",") if t],
            "tokens": [(float(t), float(n)) for t, n in [pair.split(":") for pair in tokens.split(",") if pair]]
        }

def save_usage(requests, tokens):
    with open(usage_file, "w") as f:
        f.write(",".join(map(str, requests)) + "|" + ",".join(f"{t}:{n}" for t, n in tokens))

def check_usage():
    usage = load_usage()
    now = time.time()
    
    # Clean up old requests (older than 24 hours)
    day_ago = now - 24 * 60 * 60
    usage["requests"] = [t for t in usage["requests"] if t > day_ago]
    
    # Clean up old token counts (older than 1 minute)
    minute_ago = now - 60
    usage["tokens"] = [(t, n) for t, n in usage["tokens"] if t > minute_ago]
    
    # Count requests per day
    rpd = len(usage["requests"])
    rpd_limit = int(FREE_TIER_RPD_LIMIT * WARNING_THRESHOLD)
    if rpd >= rpd_limit:
        return False, f"Approaching daily request limit ({rpd}/{FREE_TIER_RPD_LIMIT}). Stopping to stay in free tier. Try again tomorrow."
    
    # Count requests per minute
    minute_ago = now - 60
    rpm = len([t for t in usage["requests"] if t > minute_ago])
    rpm_limit = int(FREE_TIER_RPM_LIMIT * WARNING_THRESHOLD)
    if rpm >= rpm_limit:
        return False, f"Approaching minute request limit ({rpm}/{FREE_TIER_RPM_LIMIT}). Wait a minute and try again."
    
    # Count tokens per minute
    tpm = sum(n for t, n in usage["tokens"])
    tpm_limit = int(FREE_TIER_TPM_LIMIT * WARNING_THRESHOLD)
    if tpm >= tpm_limit:
        return False, f"Approaching token limit ({tpm}/{FREE_TIER_TPM_LIMIT} per minute). Wait a minute and try again."
    
    return True, (rpd, rpm, tpm)

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize BioBERT for embedding queries
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
if torch.cuda.is_available():
    model.cuda()

# Initialize Gemini and check available models
genai.configure(api_key=GEMINI_API_KEY)

# List available models to confirm free tier access
available_models = [model.name for model in genai.list_models()]
print("Available Gemini models:", available_models)

# Select a free-tier model (prefer gemini-1.5-flash-latest, fallback to gemini-1.5-pro-latest)
# Select a free-tier model (prefer gemini-pro, fallback to other available models)
preferred_model = "gemini-pro"  # Use the generally available model
if preferred_model in available_models:
    gemini_model = genai.GenerativeModel(preferred_model)
    print(f"Using model: {preferred_model}")
else:
    # Try other available models (if needed)
    for model_name in ["gemini-1.5-flash", "gemini-1.5-pro"]:
        if f"models/{model_name}" in available_models:
            gemini_model = genai.GenerativeModel(f"models/{model_name}")
            print(f"Using model: models/{model_name}")
            break  # Use the first available match
    else:
        raise ValueError("No suitable Gemini model available. Available models: " + str(available_models))

# Clean text
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    return text.strip()

# Embed text using BioBERT
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    return embedding.tolist()

# Extract text from PDF (up to 10 pages)
def extract_pdf_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    num_pages = min(len(reader.pages), 10)  # Limit to 10 pages
    text = ""
    for page in range(num_pages):
        text += reader.pages[page].extract_text() + "\n"
    return clean_text(text)

# Retrieve relevant chunks from Pinecone
def retrieve_from_pinecone(query, top_k=5):
    query_embedding = embed_text(query)
    results = index.query(
        namespace=PINECONE_NAMESPACE,
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    retrieved_chunks = [match["metadata"]["chunk"] for match in results["matches"]]
    return "\n".join(retrieved_chunks)

# Count tokens using Gemini API
def count_tokens(text):
    try:
        response = gemini_model.count_tokens(text)
        return response.total_tokens
    except exceptions.QuotaExceeded as e:
        return 0  # If quota is exceeded, return 0 to avoid counting issues

# Generate answer using Gemini
def generate_answer(query, context):
    prompt = f"""
    You are a diabetes research assistant. Answer the following question based on the provided context. If the context is insufficient, use your knowledge to provide a helpful answer, but note if the information might be limited.
    **Question**: {query}
    **Context**:
    {context}
    **Answer**:
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except exceptions.QuotaExceeded as e:
        return f"Error: Gemini API quota exceeded ({str(e)}). Try again later."
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Main function to handle user input
def diabetes_bot(query, pdf_file=None):
    # Check usage limits
    can_proceed, usage_info = check_usage()
    if not can_proceed:
        return usage_info
    
    # Step 1: Get context from PDF if uploaded
    pdf_context = ""
    if pdf_file is not None:
        pdf_context = extract_pdf_text(pdf_file)
        if pdf_context:
            pdf_context = f"Uploaded PDF content:\n{pdf_context}\n\n"

    # Step 2: Retrieve relevant chunks from Pinecone
    pinecone_context = retrieve_from_pinecone(query)
    if pinecone_context:
        pinecone_context = f"Pinecone retrieved content (latest research, 2010 onward):\n{pinecone_context}\n\n"

    # Step 3: Combine contexts
    full_context = pdf_context + pinecone_context
    if not full_context.strip():
        full_context = "No relevant context found in Pinecone or uploaded PDF."

    # Step 4: Count tokens for the prompt
    prompt = f"""
    You are a diabetes research assistant. Answer the following question based on the provided context. If the context is insufficient, use your knowledge to provide a helpful answer, but note if the information might be limited.
    **Question**: {query}
    **Context**:
    {full_context}
    **Answer**:
    """
    input_tokens = count_tokens(prompt)
    if input_tokens == 0:  # Quota exceeded during token counting
        return "Error: Gemini API quota exceeded while counting tokens. Try again later."

    # Update usage
    usage = load_usage()
    now = time.time()
    usage["requests"].append(now)
    usage["tokens"].append((now, input_tokens))
    save_usage(usage["requests"], usage["tokens"])

    # Step 5: Generate answer using Gemini
    answer = generate_answer(query, full_context)

    # Step 6: Count output tokens and update usage
    output_tokens = count_tokens(answer)
    if output_tokens == 0:  # Quota exceeded during output token counting
        return answer + "\n\nError: Gemini API quota exceeded while counting output tokens. Usage stats may be incomplete."
    usage = load_usage()
    usage["tokens"].append((now, output_tokens))
    save_usage(usage["requests"], usage["tokens"])

    # Step 7: Show usage stats
    rpd, rpm, tpm = check_usage()[1]
    usage_message = f"\n\nUsage: {rpd}/{FREE_TIER_RPD_LIMIT} requests today, {rpm}/{FREE_TIER_RPM_LIMIT} requests this minute, {tpm}/{FREE_TIER_TPM_LIMIT} tokens this minute."

    return answer + usage_message

# Gradio interface
with gr.Blocks() as app:
    gr.Markdown("""
    # Diabetes-Research-Q&A-ChatBot 🩺
    Ask questions about diabetes or upload a research paper (up to 10 pages) for Q&A.  
    **Powered by the latest diabetes research (2010 onward). For pre-2010 papers, upload your research PDF!**  
    **Running on Gemini API free tier**
    """)
    
    with gr.Row():
        query_input = gr.Textbox(label="Ask a question", placeholder="e.g., What are the latest treatments for type 2 diabetes?")
        pdf_input = gr.File(label="Upload a PDF (optional, max 10 pages)", file_types=[".pdf"])
    
    submit_button = gr.Button("Submit")
    output = gr.Textbox(label="Answer")

    submit_button.click(
        fn=diabetes_bot,
        inputs=[query_input, pdf_input],
        outputs=output
    )

# Launch the app
app.launch()