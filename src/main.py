# main.py
import streamlit as st
import os
import tempfile
from mod import process_document
from rag import DocumentRag
from ask_anything import generate_response_with_gemini, generate_summary_with_gemini
from challenge_me import generate_challenge_questions, evaluate_user_answer
import time # For simulating loading delays for better UX

# --- Streamlit UI Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Smart Assistant",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if 'document_rag' not in st.session_state:
    st.session_state.document_rag = DocumentRag(collection_name="user_document_chunks")
    print("DEBUG: DocumentRag initialized.")

if 'processed_document_chunks' not in st.session_state:
    st.session_state.processed_document_chunks = []
if 'full_document_text' not in st.session_state:
    st.session_state.full_document_text = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'challenge_questions' not in st.session_state:
    st.session_state.challenge_questions = []
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = []
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []
if 'current_panel' not in st.session_state:
    st.session_state.current_panel = 'summary' # Default panel
if 'ask_history' not in st.session_state:
    st.session_state.ask_history = [] # Stores tuples of (question, answer, justification)
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'ask_question_input_value' not in st.session_state:
    st.session_state.ask_question_input_value = ""
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False


# --- Custom CSS for basic styling and dark mode toggle ---
# This CSS will primarily target custom markdown elements and use Streamlit's theming for native widgets.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-emotion"] {
        font-family: 'Inter', sans-serif;
        /* Default text color, will be overridden by Streamlit's theme or .dark-mode */
        color: var(--text-color);
    }

    /* Define CSS variables for colors, toggled by .dark-mode class on body */
    :root {
        --background-color: #f0f2f6; /* Light mode app background */
        --text-color: #333333; /* Light mode general text */
        --sidebar-bg: #ffffff; /* Light mode sidebar background */
        --panel-bg: #ffffff; /* Light mode main panel background */
        --border-color: #e0e0e0; /* Light mode borders */
        --input-bg: #f9f9f9; /* Light mode input/textarea background */
        --chat-user-bg: #dcf8c6; /* Light mode user chat bubble */
        --chat-ai-bg: #e0e0e0; /* Light mode AI chat bubble */
        --header-bg: #ffffff; /* Light mode header background */
    }

    .dark-mode {
        --background-color: #1a1a2e; /* Dark mode app background */
        --text-color: #e0e0e0; /* Dark mode general text */
        --sidebar-bg: #0f0f1a; /* Dark mode sidebar background */
        --panel-bg: #1a1a2e; /* Dark mode main panel background */
        --border-color: #3a3a5a; /* Dark mode borders */
        --input-bg: #2a2a4a; /* Dark mode input/textarea background */
        --chat-user-bg: #4CAF50; /* Dark mode user chat bubble */
        --chat-ai-bg: #3a3a5a; /* Dark mode AI chat bubble */
        --header-bg: #1a1a2e; /* Dark mode header background */
    }

    /* Apply base colors to the Streamlit app container */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    /* Main content area styling */
    /* Use data-testid for more stable targeting */
    div[data-testid="stVerticalBlock"] > div.st-emotion-cache-18ni7ap { /* This is a common class for the main content block */
        background-color: var(--panel-bg);
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        padding: 2rem;
        margin: 1rem;
        height: calc(100vh - 2rem);
        display: flex;
        flex-direction: column;
        overflow: hidden;
        color: var(--text-color); /* Ensure text color is set for main content */
    }
    .dark-mode div[data-testid="stVerticalBlock"] > div.st-emotion-cache-18ni7ap {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3); /* Darker shadow for dark mode */
    }

    /* Sidebar styling */
    div[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--border-color);
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.05);
        border-radius: 0 1rem 1rem 0;
        color: var(--text-color); /* Ensure text color is set for sidebar */
    }
    .dark-mode div[data-testid="stSidebar"] {
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.2); /* Darker shadow for dark mode */
    }

    /* General button styling (targets Streamlit's native buttons) */
    .stButton > button {
        background-color: #4CAF50; /* Green */
        color: white;
        border-radius: 0.75rem;
        padding: 0.75rem 1.25rem;
        font-weight: 600;
        transition: background-color 0.2s, box-shadow 0.2s;
        border: none;
    }
    .stButton > button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
    }
    .stButton > button:disabled {
        background-color: #cccccc;
        cursor: not-allowed;
        box-shadow: none;
    }

    /* Sidebar navigation buttons specific styling */
    div[data-testid="stSidebarNav"] .stButton > button {
        background-color: transparent; /* Override for sidebar buttons */
        color: var(--text-color); /* Use text color variable */
        text-align: left;
        padding-left: 1rem;
    }
    div[data-testid="stSidebarNav"] .stButton > button:hover {
        background-color: #e0e0e0; /* Light hover */
        color: #333333;
    }
    .dark-mode div[data-testid="stSidebarNav"] .stButton > button:hover {
        background-color: #2a2a4a; /* Dark hover */
        color: #e0e0e0;
    }
    /* Active sidebar button styling (Streamlit's primary button style when selected) */
    div[data-testid="stSidebarNav"] .stButton > button[data-testid="stSidebarNav"] {
        background-color: #4CAF50;
        color: white;
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);
    }


    /* File uploader button styling */
    div[data-testid="stFileUploader"] button {
        background-color: #007bff;
        color: white;
    }
    div[data-testid="stFileUploader"] button:hover {
        background-color: #0056b3;
    }

    /* Chat message styling */
    .chat-message {
        padding: 10px 15px;
        border-radius: 15px;
        max-width: 70%;
        margin-bottom: 10px;
        word-wrap: break-word;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: var(--chat-user-bg);
        align-self: flex-end;
        margin-left: auto;
        color: var(--text-color);
    }
    .ai-message {
        background-color: var(--chat-ai-bg);
        align-self: flex-start;
        margin-right: auto;
        color: var(--text-color);
    }

    /* Text area and input styling */
    textarea, input[type="text"] {
        border-radius: 0.75rem;
        border: 1px solid var(--border-color);
        background-color: var(--input-bg);
        color: var(--text-color);
        padding: 0.75rem;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    textarea:focus, input[type="text"]:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.3);
        outline: none;
    }

    /* Info/Success/Error messages */
    div[data-testid="stAlert"] {
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stAlert"].info { background-color: #e0f7fa; color: #00796b; }
    div[data-testid="stAlert"].success { background-color: #e8f5e9; color: #2e7d32; }
    div[data-testid="stAlert"].error { background-color: #ffebee; color: #c62828; }
    div[data-testid="stAlert"].warning { background-color: #fffde7; color: #f9a825; }

    .dark-mode div[data-testid="stAlert"].info { background-color: #004d40; color: #80cbc4; }
    .dark-mode div[data-testid="stAlert"].success { background-color: #1b5e20; color: #a5d6a7; }
    .dark-mode div[data-testid="stAlert"].error { background-color: #b71c1c; color: #ef9a9a; }
    .dark-mode div[data-testid="stAlert"].warning { background-color: #f57f17; color: #ffee58; }

    /* For the spinner */
    .stSpinner > div > div {
        border-top-color: #4CAF50;
    }

    /* Adjust padding for the main content area to make it full width */
    div[data-testid="stVerticalBlock"] > div.st-emotion-cache-z5fcl4 {
        padding-left: 0rem;
        padding-right: 0rem;
    }
    div[data-testid="stVerticalBlock"] > div.st-emotion-cache-18ni7ap {
        padding: 2rem;
    }

    /* Fix for the sidebar expand/collapse button text */
    /* Target the button by its data-testid and hide its internal text span */
    button[data-testid="stSidebarToggle"] span {
        display: none !important;
    }
    /* Add content for the arrow symbols using ::before or ::after */
    button[data-testid="stSidebarToggle"]::before {
        content: '«'; /* Left arrow for collapse */
        font-size: 1.5rem;
        line-height: 1;
        display: inline-block;
        vertical-align: middle;
        color: var(--text-color); /* Ensure arrow color changes with theme */
    }
    /* When sidebar is expanded, change to right arrow */
    div[data-testid="stSidebar"][aria-expanded="true"] button[data-testid="stSidebarToggle"]::before {
        content: '»' !important; /* Right arrow for expand */
    }

    /* Style for the copy icon */
    .copy-icon {
        cursor: pointer;
        margin-left: 10px;
        font-size: 1.2em;
        vertical-align: middle;
        color: var(--text-color); /* Inherit text color */
        transition: color 0.2s;
    }
    .copy-icon:hover {
        color: #4CAF50; /* Green on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- JavaScript for Clipboard Copy (placed here for global availability) ---
# This script will be executed once when the app loads.
st.markdown(
    """
    <script>
    function copyTextToClipboard(text) {
        if (navigator.clipboard) { // Use modern clipboard API if available
            navigator.clipboard.writeText(text).then(function() {
                // Send message back to Streamlit to show a toast
                window.parent.postMessage({
                    streamlit: {
                        command: 'SET_PAGE_STATE',
                        state: { toast_message: 'Copied to clipboard!', toast_type: 'success' },
                        bubble: true
                    }
                }, '*');
            }, function(err) {
                window.parent.postMessage({
                    streamlit: {
                        command: 'SET_PAGE_STATE',
                        state: { toast_message: 'Failed to copy!', toast_type: 'error' },
                        bubble: true
                    }
                }, '*');
            });
        } else { // Fallback for older browsers
            var textArea = document.createElement("textarea");
            textArea.value = text;
            textArea.style.position = "fixed"; // Avoid scrolling to bottom
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            try {
                var successful = document.execCommand('copy');
                window.parent.postMessage({
                    streamlit: {
                        command: 'SET_PAGE_STATE',
                        state: { toast_message: 'Copied to clipboard (fallback)!', toast_type: 'info' },
                        bubble: true
                    }
                }, '*');
            } catch (err) {
                window.parent.postMessage({
                    streamlit: {
                        command: 'SET_PAGE_STATE',
                        state: { toast_message: 'Failed to copy (fallback)!', toast_type: 'error' },
                        bubble: true
                    }
                }, '*');
            }
            document.body.removeChild(textArea);
        }
    }

    // Listener to receive messages from the JavaScript to trigger Streamlit toasts
    window.addEventListener('message', event => {
        if (event.data.streamlit && event.data.streamlit.command === 'SET_PAGE_STATE' && event.data.streamlit.state.toast_message) {
            const message = event.data.streamlit.state.toast_message;
            const type = event.data.streamlit.state.toast_type || 'info';
            // Streamlit's toast function is available in the global scope
            if (window.streamlit && window.streamlit.setToast) {
                window.streamlit.setToast(message, type);
            }
        }
    });
    </script>
    """, unsafe_allow_html=True
)

# --- Python-side Toast Listener (to receive messages from JS) ---
# This checks for a 'toast_message' in session_state set by the JS message
if 'toast_message' in st.session_state and st.session_state.toast_message:
    toast_type = st.session_state.get('toast_type', 'info')
    if toast_type == 'success':
        st.toast(st.session_state.toast_message, icon="✅")
    elif toast_type == 'error':
        st.toast(st.session_state.toast_message, icon="❌")
    elif toast_type == 'info':
        st.toast(st.session_state.toast_message, icon="ℹ️")
    else:
        st.toast(st.session_state.toast_message)
    # Clear the toast message after displaying
    del st.session_state.toast_message
    if 'toast_type' in st.session_state:
        del st.session_state.toast_type


# --- Dark Mode Toggle (using session state and CSS class) ---
def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode
    print(f"DEBUG: Dark mode toggled to {st.session_state.dark_mode}")
    st.rerun() # Rerun to apply CSS class change

# Apply dark mode class to the body if enabled
if st.session_state.dark_mode:
    st.markdown('<script>document.body.classList.add("dark-mode");</script>', unsafe_allow_html=True)
else:
    st.markdown('<script>document.body.classList.remove("dark-mode");</script>', unsafe_allow_html=True)


# --- Sidebar UI ---
with st.sidebar:
    st.title("📚 Smart Assistant")
    st.markdown("---")

    # Dark Mode Toggle Button
    if st.button(f"{'☀️ Light Mode' if st.session_state.dark_mode else '🌙 Dark Mode'}", key="dark_mode_toggle"):
        toggle_dark_mode()
    
    st.markdown("---")

    st.header("Navigation")
    # Navigation buttons using st.button for reliability
    # Streamlit's native buttons are now used directly.
    # The active state is handled by Streamlit's internal mechanism for primary buttons.
    # We set the 'type' to 'primary' for the currently active panel's button.
    
    # Use a container for the navigation buttons to apply consistent styling
    nav_container = st.container()
    with nav_container:
        # Summary Button
        summary_button_clicked = st.button("📄 Summary", key="nav_summary", 
                                           type="primary" if st.session_state.current_panel == 'summary' else "secondary")
        if summary_button_clicked:
            if st.session_state.current_panel != 'summary':
                st.session_state.current_panel = 'summary'
                print("DEBUG: Switched to Summary panel.")
                st.rerun()

        # Ask Me Button
        ask_button_clicked = st.button("💬 Ask Me", key="nav_ask", 
                                       type="primary" if st.session_state.current_panel == 'ask' else "secondary")
        if ask_button_clicked:
            if st.session_state.current_panel != 'ask':
                st.session_state.current_panel = 'ask'
                print("DEBUG: Switched to Ask Me panel.")
                st.rerun()

        # Challenge Me Button
        challenge_button_clicked = st.button("🧠 Challenge Me", key="nav_challenge", 
                                             type="primary" if st.session_state.current_panel == 'challenge' else "secondary")
        if challenge_button_clicked:
            if st.session_state.current_panel != 'challenge':
                st.session_state.current_panel = 'challenge'
                print("DEBUG: Switched to Challenge Me panel.")
                st.rerun()


    st.markdown("---")
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload your document (PDF or TXT)", type=["pdf", "txt"], key="main_file_uploader")

    if uploaded_file:
        # Check if a new file has been uploaded or if it's a different file
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            st.success(f"Document '{uploaded_file.name}' uploaded successfully!")
            print(f"DEBUG: Uploaded file: {uploaded_file.name}")

            # Create a temporary file path and write the uploaded content to it
            fd, temp_file_path = tempfile.mkstemp(suffix=f".{uploaded_file.type.split('/')[-1]}")
            file_type = uploaded_file.type.split('/')[-1]
            if file_type == 'plain':
                file_type = 'txt'

            try:
                with os.fdopen(fd, 'wb') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                
                # Clear previous documents from ChromaDB and reset all related session states
                with st.spinner("Clearing previous data..."):
                    st.session_state.document_rag.clear_collection()
                    st.session_state.processed_document_chunks = []
                    st.session_state.full_document_text = ""
                    st.session_state.summary = ""
                    st.session_state.challenge_questions = []
                    st.session_state.user_answers = []
                    st.session_state.evaluation_results = []
                    st.session_state.ask_history = []
                    print("DEBUG: Session state cleared.")
                
                with st.spinner("Processing document and generating embeddings..."):
                    processed_chunks = process_document(temp_file_path, file_type)
                    if processed_chunks:
                        metadatas = [{"source": "uploaded_document", "chunk_index": i} for i in range(len(processed_chunks))]
                        st.session_state.document_rag.add_documents(processed_chunks, metadatas=metadatas)
                        st.session_state.processed_document_chunks = processed_chunks
                        st.session_state.full_document_text = " ".join(processed_chunks)
                        st.success("Document processed and embeddings stored!")
                        print(f"DEBUG: Processed {len(processed_chunks)} chunks.")

                        with st.spinner("Generating summary..."):
                            summary = generate_summary_with_gemini(st.session_state.full_document_text)
                            st.session_state.summary = summary
                            st.success("Summary generated!")
                            print("DEBUG: Summary generated.")
                        
                        st.session_state.current_panel = 'summary' # Switch to summary after upload
                        st.rerun() # Rerun to update main content
                    else:
                        st.error("Failed to process document. Please check the file format and content.")
                        st.session_state.uploaded_file_name = None # Reset file name on failure
                        print("ERROR: Document processing failed.")
            except Exception as e:
                st.error(f"Error saving or processing uploaded file: {e}")
                st.session_state.uploaded_file_name = None # Reset file name on failure
                print(f"ERROR: Exception during file processing: {e}")
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        else:
            st.info(f"Document '{st.session_state.uploaded_file_name}' is already loaded.")
            # If the same file is re-uploaded, just ensure the summary is displayed if available
            if st.session_state.summary and st.session_state.current_panel == 'summary':
                st.markdown(f"""
                    <div style="background-color: var(--panel-bg); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid var(--border-color); margin-top: 1rem;">
                        <h3 style="color: var(--text-color); margin-bottom: 1rem;">Current Document: {st.session_state.uploaded_file_name}</h3>
                        <p style="color: var(--text-color); white-space: pre-wrap;">{st.session_state.summary}</p>
                    </div>
                """, unsafe_allow_html=True)


# --- Main Content Panel ---
main_content_container = st.container()
with main_content_container:
    st.markdown(f"""
        <h1 style="color: var(--text-color); font-size: 2.5rem; font-weight: 700; margin-bottom: 1.5rem;">
            {st.session_state.current_panel.replace('_', ' ').title()}
        </h1>
    """, unsafe_allow_html=True)

    # Display content based on selected panel
    if not st.session_state.full_document_text:
        st.info("Please upload a document using the sidebar to begin interacting with the Smart Assistant.")
        # Add a visual cue for upload
        st.markdown("""
            <div style="text-align: center; margin-top: 50px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="color: #6c757d;">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>
                </svg>
                <p style="color: #6c757d; margin-top: 10px; font-size: 1.1rem;">Upload a document to get started!</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        if st.session_state.current_panel == 'summary':
            st.markdown(f"""
                <div style="background-color: var(--background-color); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid var(--border-color); flex-grow: 1; overflow-y: auto;">
                    <p style="color: var(--text-color); white-space: pre-wrap;">{st.session_state.summary}</p>
                </div>
            """, unsafe_allow_html=True)

        elif st.session_state.current_panel == 'ask':
            # Use a container for chat history to enable scrolling
            chat_history_container = st.container(height=400) # Fixed height for scrollable chat
            with chat_history_container:
                for q, a, j in st.session_state.ask_history:
                    st.markdown(f"""
                        <div class="chat-message user-message">
                            <p><strong>You:</strong> {q}</p>
                        </div>
                        <div class="chat-message ai-message">
                            <p><strong>Smart Assistant:</strong> {a}</p>
                            {f'<p style="font-size: 0.85em; opacity: 0.8; margin-top: 5px;"><strong>Justification:</strong> {j}</p>' if j else ''}
                        </div>
                    """, unsafe_allow_html=True)
            
            # Input for new question
            user_question = st.text_input(
                "Ask a question about the document:",
                value=st.session_state.ask_question_input_value, # Controlled input
                key="ask_question_input",
                placeholder="e.g., What are the main findings of the research?"
            )
            
            if st.button("Ask", key="ask_button"):
                if user_question.strip():
                    with st.spinner("Fetching answer..."):
                        # Retrieve relevant context from ChromaDB
                        relevant_chunks = st.session_state.document_rag.query_documents(user_question, n_results=5)
                        
                        if not relevant_chunks:
                            st.warning("No highly relevant context found for your question. The answer might be general or indicate the information isn't directly in the document.")
                            
                        # Generate response using Gemini
                        answer, justification = generate_response_with_gemini(user_question, relevant_chunks)
                        
                        st.session_state.ask_history.append((user_question, answer, justification))
                        st.session_state.ask_question_input_value = "" # Clear the input box after submission
                        st.rerun() # Rerun to update chat history
                else:
                    st.warning("Please enter a question.")

        elif st.session_state.current_panel == 'challenge':
            challenge_container = st.container(height=600) # Fixed height for scrollable challenge section
            with challenge_container:
                if st.button("Generate New Challenge Questions", key="generate_questions_button"):
                    with st.spinner("Generating challenge questions..."):
                        st.session_state.challenge_questions = generate_challenge_questions(st.session_state.full_document_text, num_questions=3)
                        st.session_state.user_answers = [""] * len(st.session_state.challenge_questions) # Initialize empty answers
                        st.session_state.evaluation_results = [None] * len(st.session_state.challenge_questions) # Initialize empty evaluation results
                    st.success("Challenge questions generated!")
                    st.rerun() # Rerun to display questions

                if not st.session_state.challenge_questions:
                    st.info("Click 'Generate New Challenge Questions' to get started.")
                else:
                    st.markdown("### Answer the following questions based on the document:")
                    
                    for i, q in enumerate(st.session_state.challenge_questions):
                        st.markdown(f"""
                            <div style="background-color: var(--background-color); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid var(--border-color); margin-bottom: 1.5rem;">
                                <p style="font-weight: 600; font-size: 1.1em; color: var(--text-color); margin-bottom: 0.75rem;">Question {i+1}: {q}</p>
                        """, unsafe_allow_html=True)

                        # Use a unique key for each text_area to prevent issues on re-render
                        st.session_state.user_answers[i] = st.text_area(
                            f"Your Answer for Q{i+1}:",
                            value=st.session_state.user_answers[i],
                            key=f"user_answer_{i}",
                            height=100,
                            placeholder="Type your answer here..."
                        )
                        
                        if st.button(f"Evaluate Answer for Q{i+1}", key=f"evaluate_btn_{i}"):
                            if st.session_state.user_answers[i].strip():
                                with st.spinner("Evaluating your answer..."):
                                    is_correct, justification, score, desired_answer_snippet = evaluate_user_answer(
                                        q, 
                                        st.session_state.user_answers[i], 
                                        st.session_state.full_document_text
                                    )
                                    evaluation_feedback = {
                                        "is_correct": is_correct,
                                        "justification": justification,
                                        "score": score,
                                        "desired_answer_snippet": desired_answer_snippet
                                    }
                                    st.session_state.evaluation_results[i] = evaluation_feedback
                                    st.success("Evaluation complete!")
                                    st.rerun() # Rerun to display evaluation results
                            else:
                                st.warning("Please provide an answer to evaluate.")
                        
                        # Display evaluation results if available
                        if st.session_state.evaluation_results[i]:
                            eval_res = st.session_state.evaluation_results[i]
                            # Ensure desired_answer_snippet is a string, even if None or empty
                            snippet_to_copy = eval_res['desired_answer_snippet'] if eval_res['desired_answer_snippet'] else ""
                            st.markdown(f"""
                                <div style="margin-top: 1rem; padding: 1rem; border-radius: 0.75rem; border: 1px solid var(--border-color); background-color: var(--background-color);">
                                    <p style="font-weight: 700; color: {'#2e7d32' if eval_res['is_correct'] else '#c62828'};">
                                        Evaluation: {'Correct' if eval_res['is_correct'] else 'Incorrect'}
                                    </p>
                                    <p style="font-weight: 600; color: var(--text-color); margin-top: 0.5rem;">Score: {eval_res['score']}/10</p>
                                    <p style="color: var(--text-color); margin-top: 0.5rem;">
                                        <strong>Justification:</strong> {eval_res['justification']}
                                    </p>
                                    {f'''
                                    <p style="color: var(--text-color); margin-top: 0.5rem; display: inline-block;">
                                        <strong>Desired Answer Snippet:</strong> {snippet_to_copy}
                                    </p>
                                    <span class="copy-icon" onclick="copyTextToClipboard(`{snippet_to_copy.replace('`', '\\`')}`)">📋</span>
                                    ''' if snippet_to_copy and snippet_to_copy != "N/A" else ''}
                                </div>
                            """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True) # End of question container
