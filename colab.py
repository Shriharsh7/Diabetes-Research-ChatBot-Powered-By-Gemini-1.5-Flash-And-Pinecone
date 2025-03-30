# Install necessary libraries
!pip install requests pinecone transformers torch

import requests
import xml.etree.ElementTree as ET
import re
import pinecone
from transformers import AutoTokenizer, AutoModel
import torch
import time
import json
from config import PINECONE_API_KEY  # Make sure to have your API key in a config file

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "diabetes-bot"
index = pc.Index(index_name)

# Check initial vector count in Pinecone (namespace 'general')
stats = index.describe_index_stats()
initial_count = stats['namespaces'].get('general', {}).get('vector_count', 0)
print(f"Initial vector count in Pinecone (namespace 'general'): {initial_count}")

# Load BioBERT
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
if torch.cuda.is_available():
    model.cuda()

# Step 1: Fetch the next 12,000 PMIDs starting from retstart=9999
def fetch_pmids(query="diabetes", max_results=12000, batch_size=9999, start_offset=9999, max_retries=3):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    all_pmids = []
    retstart = start_offset

    while len(all_pmids) < max_results:
        remaining = max_results - len(all_pmids)
        current_batch_size = min(batch_size, remaining)
        search_url = f"{base_url}esearch.fcgi?db=pubmed&term={query}&datetype=pdat&mindate=2010&sort=most+recent&retmax={current_batch_size}&retstart={retstart}&retmode=xml"
        
        # Retry logic for ESearch request
        for attempt in range(max_retries):
            try:
                response = requests.get(search_url)
                response.raise_for_status()
                # Log the raw response
                print(f"ESearch response (retstart={retstart}):")
                print(response.text)
                root = ET.fromstring(response.content)
                break
            except (requests.RequestException, ET.ParseError) as e:
                print(f"ESearch request failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print("Max retries reached. Exiting.")
                    return []
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s

        # Get total count from first request with error handling
        if retstart == start_offset:
            count_elem = root.find(".//Count")
            if count_elem is None:
                print("Error: <Count> element not found in ESearch response.")
                error_elem = root.find(".//ERROR")
                if error_elem is not None:
                    print(f"PubMed API error: {error_elem.text}")
                return []
            total_count = int(count_elem.text)
            print(f"Total matching PMIDs: {total_count}")
            if total_count == 0:
                return []

        pmids = [id_elem.text for id_elem in root.findall(".//Id")]
        all_pmids.extend(pmids)
        print(f"Fetched {len(pmids)} PMIDs (retstart={retstart}). Total PMIDs: {len(all_pmids)}")

        if len(pmids) < current_batch_size:
            print("No more PMIDs to fetch.")
            break

        retstart += len(pmids)
        time.sleep(2)  # Increased to 2-second delay to avoid rate limiting

    return all_pmids[:max_results]

# Step 2: Fetch abstracts in batches of 100 with 0.1-second delay
def fetch_abstracts(pmids, batch_size=100):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    abstracts = []
    counter = 1  # Fallback for missing PMIDs

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={','.join(batch)}&retmode=xml"
        response = requests.get(fetch_url)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        for article in root.findall(".//Article"):
            pmid = article.find(".//PMID").text if article.find(".//PMID") is not None else None
            title = article.find(".//ArticleTitle").text if article.find(".//ArticleTitle") is not None else ""
            abstract = article.find(".//AbstractText")
            abstract_text = abstract.text if abstract is not None else ""

            if abstract_text and pmid:
                abstracts.append({"pmid": pmid, "title": title, "abstract": abstract_text})
            elif abstract_text:
                abstracts.append({"pmid": f"abstract_{counter}", "title": title, "abstract": abstract_text})
                counter += 1

        # Save progress
        with open("abstracts_progress.json", "a") as f:
            json.dump(abstracts[-len(batch):], f)
            f.write("\n")
        print(f"Fetched {len(abstracts)} abstracts so far...")
        time.sleep(0.1)  # 0.1-second delay

    return abstracts

# Step 3: Clean text
def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    return text.strip()

# Step 4: Chunk text into 1000-character pieces
def chunk_text(text, max_length=1000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word) + 1  # +1 for space
        if current_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Step 5: Embed text in batches
def embed_text_batch(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeddings)
    return embeddings

# Step 6: Upsert to Pinecone
def upsert_to_pinecone(abstracts, namespace="general", batch_size=100):
    vectors = []
    
    # Clean, filter, and chunk
    for entry in abstracts:
        clean_abstract = clean_text(entry["abstract"])
        clean_title = clean_text(entry["title"])
        if not clean_abstract:  # Skip if no abstract after cleaning
            continue
        chunks = chunk_text(clean_abstract, max_length=1000)
        
        for chunk_idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            vector_id = f"{entry['pmid']}_{chunk_idx}"
            vectors.append({
                "id": vector_id,
                "values": None,  # Placeholder for embedding
                "metadata": {
                    "title": clean_title,
                    "abstract": clean_abstract,
                    "chunk": chunk,
                    "chunk_idx": chunk_idx
                }
            })

    print(f"Total chunks to process: {len(vectors)}")

    # Embed all chunks
    chunk_texts = [v["metadata"]["chunk"] for v in vectors]
    embeddings = embed_text_batch(chunk_texts)
    for i, embedding in enumerate(embeddings):
        vectors[i]["values"] = embedding.tolist()

    # Upsert in batches
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        response = index.upsert(vectors=batch, namespace=namespace)
        print(f"Upserted batch {i//batch_size + 1}: {i+len(batch)}/{len(vectors)} chunks. Response: {response}")

    stats = index.describe_index_stats()
    vector_count = stats['namespaces'].get(namespace, {}).get('vector_count', 0)
    print(f"Total vectors in Pinecone (namespace '{namespace}'): {vector_count}")

# Main execution
if __name__ == "__main__":
    print("Fetching the next 12,000 PMIDs, bro!")
    
    # Fetch PMIDs starting from retstart=9999
    pmids = fetch_pmids()
    print(f"Fetched {len(pmids)} PMIDs")

    # Fetch abstracts
    abstracts = fetch_abstracts(pmids)
    print(f"Fetched {len(abstracts)} abstracts")

    # Process and upsert to Pinecone
    upsert_to_pinecone(abstracts)
    print("All done, Upserted to Pinecone!")