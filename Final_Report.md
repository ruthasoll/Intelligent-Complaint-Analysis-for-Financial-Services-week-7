# Intelligent Complaint Analysis Data Product: Final Report

## 1. Introduction: Turning Feedback into Strategy
At CrediTrust Financial, customer feedback is abundant but often untapped. With thousands of complaints pouring in monthly across multiple channels, extracting actionable insights has surprisingly remained a manual, bottlenecked process. Product Managers like Asha spend hours reading individual narratives, often missing broader trends hidden in the noise.

This project introduces an **Intelligent Complaint Analysis Agent**: a Retrieval-Augmented Generation (RAG) system designed to democratize data access. By allowing team members to ask plain-English questions—*"What are the recurring issues with virtual currency transfers?"*—and receiving instantly synthesized, evidence-backed answers, we are shifting CrediTrust from reactive support to proactive product improvement.

## 2. Technical Architecture & Implementation

### 2.1 The Data Strategy
We utilized the Consumer Financial Protection Bureau (CFPB) dataset, filtering for five key financial products. 
*   **Data Cleaning**: We removed ~70% of records that lacked narrative descriptions and normalized the text to reduce noise.
*   **Stratified Sampling**: To ensure fair representation across products (avoiding a bias toward Credit Cards), we employed stratified sampling to select a balanced corpus of ~12,000 complaints.

### 2.2 The Semantic Search Engine (Task 2)
Traditional keyword search fails when users ask concept-based questions. We built a semantic search engine using:
*   **Chunking**: Recursive splitting (500 chars, 50 overlap) to preserve narrative context.
*   **Embeddings**: The `sentence-transformers/all-MiniLM-L6-v2` model, chosen for its efficiency/performance ratio on the MTEB benchmark.
*   **Vector Store**: **ChromaDB** serves as our vector database, storing embeddings alongside rich metadata (product, issue, complaint ID) to enable precise retrieval and source attribution.

### 2.3 The RAG Core (Task 3)
The intelligence layer combines retrieval with generative AI:
*   **Retriever**: Fetches the top 3 most relevant context chunks for a given query.
*   **LLM**: We integrated **Google's Flan-T5-Base**. This model was selected for its strong instruction-following capabilities relative to its size, allowing for deployment in resource-constrained environments (CPU inference).
*   **Prompt Engineering**: We designed a strict prompt template that forces the model to answer *only* based on retrieved context, reducing the risk of hallucinations.

### 2.4 The Interactive Interface (Task 4)
To make this tool accessible to non-technical staff, we built a web interface using **Gradio**.
*   **Key Features**:
    *   Simple Q&A Chat Layout.
    *   **Transparent Citations**: Every answer includes an expandable "Retrieved Sources" section, showing the exact complaint text and ID used to generate the response. This feature is critical for building trust with Compliance/Risk teams.

## 3. System Evaluation

We evaluated the system against a set of "Gold Standard" questions. Below is a summary of the performance:

| Question | Generated Answer Quality | Retrieval Relevance | Analysis |
| :--- | :--- | :--- | :--- |
| *What are the common fraud issues with Credit Cards?* | ⭐⭐ (2/5) | ⭐⭐⭐⭐ (4/5) | Retrieval found relevant card complaints, but the LLM was overly cautious ("I don't have enough info") due to strict prompting. |
| *How do customers describe problems with money transfers?* | ⭐⭐⭐⭐ (4/5) | ⭐⭐⭐⭐⭐ (5/5) | Excellent result. The system synthesized issues regarding "frozen accounts" and "reversals" accurately from the context. |
| *Why do customers complain about student loans?* | ⭐⭐ (2/5) | ⭐⭐⭐ (3/5) | Retrieved chunks were specific to individual loan servicing but the model struggled to generalize a high-level "why". |

**Key Finding**: The `Flan-T5-Base` model is robust but has a small context window (512 tokens). This required aggressive truncation of retrieved contexts, sometimes causing the model to miss key details present in the vectors.

## 4. UI Showcase
*(Please insert screenshots of the Gradio application here)*

## 5. Conclusion & Future Improvements

### Challenges & Learnings
*   **Context Window Constraints**: Working with local, smaller LLMs requires satisfying strict token limits, which clashes with the need for broad context in RAG.
*   **Data Imbalance**: Even with stratified sampling, certain nuanced issues (e.g., specific fraud types) are rare and harder to retrieve.

### Next Steps
1.  **Model Upgrade**: Switch to a larger model like **Mistral-7B-Quantized** or **Llama-3-8B** to utilize a larger context window (4k+ tokens) for more comprehensive answers.
2.  **Hybrid Search**: Implement a hybrid retriever (Keyword + Vector) to better handle specific terminology.
3.  **Deployment**: Dockerize the application for easy internal distribution.
