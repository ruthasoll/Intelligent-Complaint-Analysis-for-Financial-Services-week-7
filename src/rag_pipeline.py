
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Fix paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, 'vector_store')

def load_vectorstore():
    print(f"Loading vector store from {VECTOR_STORE_PATH}...")
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(f"Vector store not found at {VECTOR_STORE_PATH}. Run Task 2 first.")
        
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embedding_model)
    return vectorstore

def setup_llm():
    print("Loading LLM (google/flan-t5-base)...")
    model_id = "google/flan-t5-base"
    
    # Check for GPU
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device==0 else 'CPU'}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=512,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.15,
        device=device
    )
    
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

def build_rag_pipeline(vectorstore, llm):
    # Returning the components needed for manual execution
    return {"vectorstore": vectorstore, "llm": llm}

def evaluate_rag(pipeline_components):
    print("\n--- Starting Qualitative Evaluation ---")
    vectorstore = pipeline_components["vectorstore"]
    llm = pipeline_components["llm"]
    
    test_questions = [
        "What are the common fraud issues with Credit Cards?",
        "Why do customers complain about student loans?",
        "What are the main issues with checking accounts?",
        "How do customers describe problems with money transfers?",
        "Are there any complaints about customer service being rude?"
    ]
    
    # Reduced k to 3 to fit within Flan-T5's 512 token limit
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    results = []
    
    for q in test_questions:
        print(f"\nQuestion: {q}")
        try:
            # 1. Retrieve
            docs = retriever.invoke(q)
            
            # 2. Format Context (with character limit)
            # Flan-T5 has ~512 token limit. 1 token ~= 4 chars. Max chars ~2000.
            # Reserve ~100 chars for prompt/question.
            # Truncating context to 1800 chars.
            full_context = "\n\n".join([doc.page_content for doc in docs])
            context = full_context[:1800]
            
            # 3. Create Prompt
            prompt = f"""You are a helpful financial analyst assistant for CrediTrust. 
Answer the question based ONLY on the following context. 
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {q}

Answer:"""

            # 4. Generate
            # HuggingFacePipeline is a langchain object, so we can call invoke or predict?
            # Or if it's a Runnable, invoke.
            # Local LLM is wrapped in HuggingFacePipeline.
            answer = llm.invoke(prompt)

            sources = []
            for doc in docs:
                meta = doc.metadata
                sources.append(f"{meta.get('product', 'Unknown')} (ID: {meta.get('complaint_id', 'N/A')})")
            
            print(f"Answer: {answer}")
            print(f"Sources: {sources[:2]}") # Show top 2 sources
            
            results.append({
                "Question": q,
                "Answer": answer,
                "Sources": sources
            })
        except Exception as e:
            print(f"Error processing question '{q}': {e}")
            import traceback
            traceback.print_exc()
            
    return results

if __name__ == "__main__":
    if not os.path.exists(VECTOR_STORE_PATH):
        print("Vector store does not exist. Please run Task 2 embedding pipeline first.")
    else:
        try:
            vs = load_vectorstore()
            llm = setup_llm()
            pipeline_comps = build_rag_pipeline(vs, llm)
            evaluate_rag(pipeline_comps)
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
