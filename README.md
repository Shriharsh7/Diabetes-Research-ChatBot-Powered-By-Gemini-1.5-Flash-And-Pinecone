# Diabetes Research ChatBot Powered By Gemini 1.5 Flash And Pinecone     

Welcome to the **Diabetes Research Q&A Chatbot**—an AI-powered tool that transforms how we access and understand diabetes research. This project blends cutting-edge natural language processing (NLP), vector search, and generative AI to deliver fast, precise, and context-aware answers to medical queries. Hosted on Hugging Face Spaces, it’s built from scratch using free-tier tools, proving that innovation doesn’t need a big budget.

[**Try It Live**](https://huggingface.co/spaces/Shriharsh/Diabetes_Research_ChatBot) – Jump in and explore diabetes research like never before!

---

## What It Does:    

This chatbot is designed to make diabetes research accessible to everyone—researchers, clinicians, or curious minds. It draws from a robust dataset of **9,999 PubMed abstracts (2010 onward)** and lets users upload PDFs (up to 10 pages) for extra context. Powered by **BioBERT** for medical-grade embeddings, **Pinecone** for rapid vector retrieval, and **Google’s Gemini 1.5 Flash** for smart answer generation, it delivers responses in **8-9 seconds**. Built at zero cost, it’s a scalable, efficient solution that showcases advanced AI engineering.

---

## Key Features:

- **Extensive Research Database**:  
  Harvested **9,999 PubMed abstracts** focused on diabetes, forming a comprehensive knowledge base of the latest studies.

- **Medical-Grade Embeddings**:  
  Leverages **BioBERT** to create **768-dimensional vectors**, capturing the nuances of biomedical text for accurate retrieval.

- **Lightning-Fast Retrieval**:  
  Precomputed embeddings are statically upserted into **Pinecone**, slashing latency and enabling scalable lightening fast vector searches.

- **Dynamic PDF Integration**:  
  Upload a PDF, and its text is extracted and embedded in real-time with BioBERT, enriching the chatbot’s context.

- **Intelligent Answer Generation**:  
  Retrieved data feeds into **Google’s Gemini 1.5 Flash**, which crafts concise, readable, and context-driven answers—falling back to its own knowledge if needed.

- **Low Latency**:  
  Optimized pipeline delivers answers in **8-9 seconds**, balancing speed and accuracy.

- **Zero-Cost Innovation**:  
  Engineered entirely on free-tier resources—Google Colab’s T4 GPU, Pinecone’s free tier, and Gemini’s API—demonstrating resourcefulness and scalability.

---

## Tech Stack:

Here’s the toolkit that powers this project, chosen for performance and synergy:

- **BioBERT (`dmis-lab/biobert-base-cased-v1.1`)**:  
  A BERT model fine-tuned on biomedical data, ideal for generating precise embeddings from medical text.

- **Pinecone**:  
  A managed vector database that stores embeddings for rapid similarity searches with minimal overhead.

- **Google’s Gemini (`1.5 Flash`)**:  
  A generative AI model that produces high-quality, human-like answers from complex contexts.

- **Gradio**:  
  Drives a sleek, user-friendly interface hosted on Hugging Face Spaces.

- **PyPDF2**:  
  Extracts text from uploaded PDFs, enabling dynamic context expansion.

- **Python Ecosystem**:  
  Relies on `torch`, `transformers`, `requests`, and more for seamless integration and processing.

---

## How It Works:

The chatbot’s workflow is a streamlined, optimized pipeline:

1. **Data Collection**:  
   - Fetched **9,999 PubMed abstracts** via the PubMed API, targeting diabetes research from 2010+.  
   - Cleaned text (removing HTML, non-ASCII characters) and chunked it into **1000-character segments** for efficient embedding.

2. **Embedding Pipeline**:  
   - Used **BioBERT** on Google Colab’s free T4 GPU to generate **768-dimensional embeddings**.  
   - Batched and upserted embeddings into **Pinecone** as a static knowledge store, minimizing runtime computation.

3. **Query Processing**:  
   - User queries are embedded with BioBERT and searched against Pinecone for the top-k relevant chunks.  
   - Uploaded PDFs (if any) are processed with PyPDF2 and BioBERT, adding real-time context.

4. **Answer Synthesis**:  
   - Retrieved context (from Pinecone and PDFs) is fed to **Google’s Gemini 1.5 Flash**, which generates a polished, context-aware response.  
   - If the context is thin, Gemini taps its broader knowledge to ensure a helpful answer.

5. **Resource Management**:  
   - Tracks API usage (Pinecone, Gemini) with exponential backoff, staying within free-tier limits without hiccups.

This architecture ensures speed, precision, and scalability—all at zero cost.

---

## Engineering Highlights:

- **Precision with BioBERT**:  
  The 768-dimensional embeddings capture medical nuances, making retrieval highly relevant.

- **Efficiency via Static Upsert**:  
  Precomputing and storing embeddings in Pinecone cuts query time dramatically.

---
## Challenges and Solutions:

Building this wasn’t a walk in the park—here’s how I tackled the tough stuff:

- **Large-Scale Data Processing**:  
  - **Challenge**: Embedding 9,999 abstracts on free-tier resources.  
  - **Solution**: Batched processing on Colab’s T4 GPU and chunked upserts to Pinecone.

- **Latency Optimization**:  
  - **Challenge**: Keeping responses fast with multiple API calls.  
  - **Solution**: Precomputed embeddings and lean server-side logic on Hugging Face Spaces.

- **API Rate Limits**:  
  - **Challenge**: Avoiding bottlenecks with PubMed, Pinecone, and Gemini.  
  - **Solution**: Added usage tracking and exponential backoff for graceful retries.

- **PDF Handling**:  
  - **Challenge**: Real-time PDF processing without slowdowns.  
  - **Solution**: Streamlined extraction with PyPDF2 and quick embeddings with BioBERT.

These fixes showcase my ability to optimize AI workflows under constraints—a skill I’m proud to bring to the table.

---

## Future Directions

This is just the beginning—here’s where it could go:

- **Expanded Scope**:  
  Include full-text articles or other fields like oncology or cardiology or even scale to 1 Million articles or all the peer reviewed published diabetes articles till date.

- **Real-Time Updates**:  
  Automate PubMed fetches to keep the dataset current.

- **Advanced PDF Support**:  
  Add OCR for scanned docs and handle larger files.

- **Multilingual Queries**:  
  Integrate translation for global accessibility.

- **Visual Enhancements**:  
  Add charts or paper links alongside answers for richer insights.

- **Enterprise Scaling**:  
  Transition to cloud infrastructure for broader deployment.

---

## How to Use It

1. **Visit the App**:  
   Go to the [Hugging Face Space](https://huggingface.co/spaces/Shriharsh/Diabetes_Research_ChatBot).  
2. **Ask Away**:  
   Try something like “What’s new in type 2 diabetes treatments?”  
3. **Add a PDF (Optional)**:  
   Upload a research paper for tailored context.  
4. **Get Your Answer**:  
   Submit and see the response in 8-9 seconds.

No setup, no fuss—just instant access to diabetes research.

---

## Acknowledgments

This project owes a lot to some amazing free resources:

- **PubMed**: For the treasure trove of research data.  
- **BioBERT**: For its biomedical NLP prowess.  
- **Pinecone**: For fast, free vector storage.  
- **Google Colab**: For the T4 GPU that made embedding possible.  
- **Google’s Gemini 1.5 Flash**: For smart, readable answers.  
- **Hugging Face Spaces & Gradio**: For a slick, hosted UI.

---

## Why It Matters and Conclusion:

The Diabetes Research Q&A Chatbot is more than a tool—it’s a proof of concept. It shows how to fuse **BioBERT**, **Pinecone**, and **Google’s Gemini 1.5 Flash** into a system that’s fast, accurate, and accessible. The Diabetes Research Chatbot is a cutting-edge tool that exemplifies the fusion of advanced natural language processing, vector search technology, and generative AI. Designed to serve both researchers and clinicians, it offers unparalleled insights into the latest diabetes research while operating with remarkable efficiency and scalability. This project is not only a demonstration of technical prowess but also a showcase of innovative thinking in the realm of medical AI.
Building it with free tools was a fun challenge, and I learned a ton about juggling APIs and optimizing pipelines. It’s not perfect, but it’s a solid start, and I’m excited to see where I can take it next!

*Built with passion by Shriharsh*  
Feel free to explore, fork, or reach out—I’d love to talk about how this can grow!
