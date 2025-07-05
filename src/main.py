# main.py
import streamlit as st
import os
import tempfile
from mod import process_document
from rag import DocumentRag
from ask_anything import generate_response_with_gemini, generate_summary_with_gemini
from challenge_me import generate_challenge_questions, evaluate_user_answer

# Initialize DocumentRag (using a session-specific collection name for isolation)
# Using st.session_state to persist the rag across reruns
if 'document_rag' not in st.session_state:
    st.session_state.document_rag = DocumentRag(collection_name="user_document_chunks")
    print("DocumentRag initialized in session state.")

# Streamlit UI
st.set_page_config(layout="wide", page_title="EZ Smart Assistant")

st.title("📚 EZ Smart Assistant for Research Summarization")
st.markdown("Upload a document (PDF or TXT) to get started. Ask questions or challenge your understanding!")

# File Uploader
uploaded_file = st.file_uploader("Upload your document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name
        file_type = uploaded_file.type.split('/')[-1] # 'pdf' or 'txt'

    st.success(f"Document '{uploaded_file.name}' uploaded successfully!")

    # Process document only if it's new or has changed
    if 'last_uploaded_file_name' not in st.session_state or st.session_state.last_uploaded_file_name != uploaded_file.name:
        st.session_state.last_uploaded_file_name = uploaded_file.name
        
        # Clear previous documents from ChromaDB
        st.session_state.document_rag.clear_collection()
        st.session_state.processed_document_chunks = []
        st.session_state.full_document_text = ""
        st.session_state.summary = ""
        st.session_state.challenge_questions = [] # Clear challenge questions on new upload
        st.session_state.user_answers = [] # Clear user answers
        st.session_state.evaluation_results = [] # Clear evaluation results

        with st.spinner("Processing document and generating embeddings..."):
            processed_chunks = process_document(file_path, file_type)
            if processed_chunks:
                # Pass metadatas explicitly to satisfy ChromaDB's requirement
                # Each chunk will have a default metadata {"source": "uploaded_document"}
                metadatas = [{"source": "uploaded_document", "chunk_index": i} for i in range(len(processed_chunks))]
                st.session_state.document_rag.add_documents(processed_chunks, metadatas=metadatas) # <--- UPDATED CALL
                st.session_state.processed_document_chunks = processed_chunks
                st.session_state.full_document_text = " ".join(processed_chunks) # Store full text for summary/challenge
                st.success("Document processed and embeddings stored!")

                # Generate and display summary
                with st.spinner("Generating summary..."):
                    summary = generate_summary_with_gemini(st.session_state.full_document_text)
                    st.session_state.summary = summary
                    st.subheader("Document Summary (≤ 150 words)")
                    st.info(summary)
            else:
                st.error("Failed to process document. Please check the file format and content.")
    else:
        # If same file re-uploaded, just show existing summary
        if st.session_state.summary:
            st.subheader("Document Summary (≤ 150 words)")
            st.info(st.session_state.summary)

    # Clean up the temporary file
    os.unlink(file_path)

    # Interaction Modes
    if st.session_state.processed_document_chunks:
        st.subheader("Choose Interaction Mode")
        mode = st.radio("Select a mode:", ("Ask Anything", "Challenge Me"), horizontal=True)

        if mode == "Ask Anything":
            st.markdown("---")
            st.subheader("Ask Anything Mode")
            user_question = st.text_input("Ask a question about the document:")

            if user_question:
                with st.spinner("Fetching answer..."):
                    # Retrieve relevant context from ChromaDB
                    relevant_chunks = st.session_state.document_rag.query_documents(user_question, n_results=5)
                    
                    if not relevant_chunks:
                        st.warning("No highly relevant context found for your question. The answer might be general or indicate the information isn't directly in the document.")
                        
                    # Generate response using Gemini
                    answer, justification = generate_response_with_gemini(user_question, relevant_chunks)
                    
                    st.markdown(f"**Answer:** {answer}")
                    if justification:
                        st.markdown(f"**Justification:** __{justification}__")
                    else:
                        st.markdown(f"**Justification:** __{'No specific justification found in the AI response.'}__")

        elif mode == "Challenge Me":
            st.markdown("---")
            st.subheader("Challenge Me Mode")

            # Initialize challenge_questions and user_answers if not present or empty
            if 'challenge_questions' not in st.session_state or not st.session_state.challenge_questions:
                st.session_state.challenge_questions = []
                st.session_state.user_answers = []
                st.session_state.evaluation_results = []

            if st.button("Generate New Challenge Questions"):
                with st.spinner("Generating challenge questions..."):
                    st.session_state.challenge_questions = generate_challenge_questions(st.session_state.full_document_text, num_questions=3)
                    st.session_state.user_answers = [""] * len(st.session_state.challenge_questions) # Initialize empty answers
                    st.session_state.evaluation_results = [""] * len(st.session_state.challenge_questions) # Initialize empty evaluation results
            
            if st.session_state.challenge_questions:
                st.markdown("Answer the following questions based on the document:")
                
                for i, q in enumerate(st.session_state.challenge_questions):
                    st.markdown(f"**Question {i+1}:** {q}")
                    
                    # Use a unique key for each text_area to prevent issues on re-render
                    st.session_state.user_answers[i] = st.text_area(f"Your Answer for Q{i+1}:", value=st.session_state.user_answers[i], key=f"user_answer_{i}")
                    
                    if st.button(f"Evaluate Answer for Q{i+1}", key=f"evaluate_btn_{i}"):
                        with st.spinner("Evaluating your answer..."):
                            is_correct, justification = evaluate_user_answer(
                                q, 
                                st.session_state.user_answers[i], 
                                st.session_state.full_document_text
                            )
                            evaluation_feedback = f"**Evaluation:** {'Correct' if is_correct else 'Incorrect'}\n**Justification:** {justification}"
                            st.session_state.evaluation_results[i] = evaluation_feedback
                            st.success("Evaluation complete!")
                    
                    if i < len(st.session_state.evaluation_results) and st.session_state.evaluation_results[i]:
                        st.markdown(st.session_state.evaluation_results[i])
                    st.markdown("---")
            else:
                st.info("Click 'Generate New Challenge Questions' to get started.")

else:
    st.info("Please upload a document to begin interacting with the Smart Assistant.")
