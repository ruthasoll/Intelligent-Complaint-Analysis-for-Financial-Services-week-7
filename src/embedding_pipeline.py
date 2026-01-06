
import os
import certifi
# Fix for potential SSL cert path issues in this environment
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import pandas as pd
import chromadb
# from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sklearn.model_selection import train_test_split

# Adjust paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'filtered_complaints.csv')
VECTOR_STORE_PATH = os.path.join(BASE_DIR, 'vector_store')

def run_pipeline():
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Ensure necessary columns
    required_cols = ['Consumer complaint narrative', 'Product']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns. Found: {df.columns}")
        return

    print(f"Total records loaded: {len(df)}")
    
    # Stratified Sampling (10k-15k)
    target_sample_size = 12000 # Aiming for ~12k
    
    if len(df) > target_sample_size:
        print(f"Performing stratified sampling to reduce to ~{target_sample_size} records...")
        try:
            # Drop NaN in Product just in case, though Task 1 should have handled it
            df = df.dropna(subset=['Product'])
            
            # Stratify
            # If a class has fewer members than n_splits, train_test_split might complain if we aren't careful, 
            # but for a simple resize, we can use stratify. 
            # If any class has < 2 samples, stratify fails. Task 1 filtered major products, so should be fine.
            
            _, sample_df = train_test_split(
                df, 
                test_size=target_sample_size, 
                stratify=df['Product'],
                random_state=42
            )
        except Exception as e:
            print(f"Sampling warning (falling back to random sample): {e}")
            sample_df = df.sample(n=target_sample_size, random_state=42)
    else:
        sample_df = df

    print(f"Sampled records: {len(sample_df)}")
    print("Sample distribution by Product:")
    print(sample_df['Product'].value_counts())

    # Chunking
    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    documents = []
    # Using itertuples for slight speedup over iterrows
    for row in sample_df.itertuples(index=True):
        text = getattr(row, 'Consumer_complaint_narrative', None)
        # Handle column name with spaces if itertuples replaces them? 
        # Pandas itertuples creates namedtuple with valid identifiers. 'Consumer complaint narrative' -> '_1' or custom?
        # Actually it generally replaces spaces with underscores. Let's check or stick to iterrows for safety with column names containing spaces.
        pass

    # Revert to iterrows for safety with arbitrary column names
    for idx, row in sample_df.iterrows():
        text = row['Consumer complaint narrative']
        if pd.isna(text):
            continue
            
        # Metadata
        # We need to handle potential float/nan issues in metadata for Chroma
        def clean_meta(val):
            return str(val) if pd.notna(val) else ""

        metadata = {
            "complaint_id": clean_meta(row.get('Complaint ID', idx)), 
            "product": clean_meta(row['Product']),
            "sub_product": clean_meta(row.get('Sub-product')),
            "issue": clean_meta(row.get('Issue')),
            "sub_issue": clean_meta(row.get('Sub-issue'))
        }
        
        chunks = text_splitter.split_text(str(text))
        for i, chunk in enumerate(chunks):
            doc_meta = metadata.copy()
            doc_meta['chunk_index'] = i
            documents.append(Document(page_content=chunk, metadata=doc_meta))

    print(f"Total chunks created: {len(documents)}")

    # Embedding and Indexing
    print("Initializing Vector Store (ChromaDB)...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    print(f"Creating vector store at {VECTOR_STORE_PATH}...")
    # This might take a while
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=VECTOR_STORE_PATH
    )
    
    print("Vector store created successfully.")

if __name__ == "__main__":
    run_pipeline()
