
import gradio as gr
from src.rag_pipeline import load_vectorstore, setup_llm, build_rag_pipeline
import os

# Initialize components globally
print("Initializing app components...")
try:
    vectorstore = load_vectorstore()
    llm = setup_llm()
    pipeline_comps = build_rag_pipeline(vectorstore, llm)
    retriever = pipeline_comps["vectorstore"].as_retriever(search_kwargs={"k": 3})
    generator = pipeline_comps["llm"]
    print("Initialization complete.")
except Exception as e:
    print(f"Error initializing app: {e}")
    retriever = None
    generator = None

def get_answer(question):
    if not retriever or not generator:
        return "System not initialized correctly. Please check server logs.", ""
    
    try:
        # Retrieve
        docs = retriever.invoke(question)
        
        # Context
        full_context = "\n\n".join([doc.page_content for doc in docs])
        context = full_context[:1800] # Truncate for Flan-T5
        
        # Prompt
        prompt = f"""You are a helpful financial analyst assistant for CrediTrust. 
Answer the question based ONLY on the following context. 
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
        
        # Generate
        answer = generator.invoke(prompt)
        
        # Format sources
        sources_text = ""
        for i, doc in enumerate(docs):
            meta = doc.metadata
            sources_text += f"**Source {i+1}** (ID: {meta.get('complaint_id', 'N/A')}, Product: {meta.get('product', 'N/A')}):\n"
            sources_text += f"_{doc.page_content[:200]}..._\n\n"
            
        return answer, sources_text
        
    except Exception as e:
        return f"Error generation answer: {str(e)}", ""

# Define Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # CrediTrust Financial Complaint Assistant
        Ask questions about customer complaints and get AI-generated insights.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            question_input = gr.Textbox(
                label="Your Question", 
                placeholder="e.g., Why do customers complain about wire transfers?"
            )
            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear")
        
    with gr.Row():
        with gr.Column():
            answer_output = gr.Markdown(label="Answer")
            
    with gr.Row():
        with gr.Accordion("Retrieved Sources", open=False):
            sources_output = gr.Markdown()
            
    # Examples
    gr.Examples(
        examples=[
            "What are the common fraud issues with Credit Cards?",
            "Why do customers complain about student loans?",
            "How do customers describe problems with money transfers?",
            "Are there complaints about account closures?"
        ],
        inputs=question_input,
        label="Try these examples:"
    )

    submit_btn.click(
        fn=get_answer,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )
    
    question_input.submit(
        fn=get_answer,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )
    
    # Clear button logic
    def clear_fields():
        return "", "", ""
        
    clear_btn.click(
        fn=clear_fields,
        inputs=None,
        outputs=[question_input, answer_output, sources_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)
