# ask_anything.py
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
from typing import Tuple

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please set it.")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

def generate_response_with_gemini(
    question: str,
    context_chunks: list[str],
    max_output_tokens: int = 500
) -> Tuple[str, str]:
    """
    Generates a response to a question using the Gemini API,
    grounded in the provided context.

    Args:
        question (str): The user's question.
        context_chunks (list[str]): A list of relevant text chunks from the document.
        max_output_tokens (int): Maximum number of tokens for the generated response.

    Returns:
        Tuple[str, str]: A tuple containing the generated answer and its justification.
                         Returns empty strings if generation fails.
    """
    if not context_chunks:
        return "I cannot answer this question as no relevant context was found in the document.", ""

    # Combine context chunks into a single string, labeling them for the model's reference
    formatted_context = "\n\n".join([f"Document Chunk [{i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])

    # Construct the prompt for Gemini
    # Emphasize grounding the answer in the provided context and providing the exact supporting text.
    prompt = f"""
    You are an AI assistant designed to answer questions based *only* on the provided document chunks.
    If the question cannot be answered from the context, state that you cannot answer it.
    
    Provide your Answer first. Ensure the Answer is concise and does NOT include any explicit references to document chunks or "new line" type phrases.
    
    Then, on a *separate new line*, precisely beginning with "Justification (Reference text from the document):", provide the *exact text snippets* from the Document Chunk(s) that directly support your answer. If multiple chunks support it, combine the relevant parts. **Do not just reference chunk numbers; provide the actual text.**

    Document Chunks:
    {formatted_context}

    Question: {question}

    Answer:
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2, # Lower temperature for more deterministic answers
                max_output_tokens=max_output_tokens,
            )
        )
        full_response_content = response.text.strip()

        answer_text = ""
        justification = "No specific justification found in the AI response." # Default justification

        # Find the start of "Justification (Reference text from the document):" case-insensitively
        justification_marker = "justification (reference text from the document):"
        justification_idx = full_response_content.lower().find(justification_marker)

        if justification_idx != -1:
            # If the justification marker is found, everything before it is the answer
            # and everything after is the justification.
            answer_part = full_response_content[:justification_idx].strip()
            justification_part = full_response_content[justification_idx + len(justification_marker):].strip()
            
            # Clean "Answer:" prefix from answer_part
            if answer_part.lower().startswith("answer:"):
                answer_text = answer_part[len("answer:"):].strip()
            else:
                answer_text = answer_part

            justification = justification_part
        else:
            # If justification marker is not found, assume the whole response is the answer
            # and try to extract a context reference from it (as a fallback)
            raw_answer_content = full_response_content
            if raw_answer_content.lower().startswith("answer:"):
                raw_answer_content = raw_answer_content[len("answer:"):].strip()
            
            answer_text = raw_answer_content # Default answer text

            # Fallback: Look for patterns like "(Document Chunk [X])" if model didn't follow instructions
            context_ref_patterns = [
                r'\(Document Chunk \[\d+\](?: and Document Chunk \[\d+\])*\)',
                r'\(Context \[\d+\](?: and Context \[\d+\])*\)' 
            ]
            
            for pattern in context_ref_patterns:
                context_ref_in_answer_match = re.search(pattern, answer_text, re.IGNORECASE)
                if context_ref_in_answer_match:
                    extracted_ref = context_ref_in_answer_match.group(0)
                    justification = f"This is supported by {extracted_ref.strip('()')}"
                    answer_text = answer_text.replace(extracted_ref, "").strip()
                    if answer_text.endswith('.'):
                        answer_text = answer_text[:-1].strip()
                    break 

        # Final cleanup for any unwanted phrases in the answer text
        answer_text = answer_text.replace("new line", "").strip() 
        # Also remove any leftover "This is supported by Document Chunk [X]" if model put it despite instructions
        answer_text = re.sub(r'\s*This is supported by Document Chunk \[.*?\]\.\s*$', '', answer_text, flags=re.IGNORECASE)


        return answer_text, justification

    except Exception as e:
        print(f"Error generating content with Gemini: {e}")
        return "An error occurred while trying to generate a response. Please try again.", "No specific justification found in the AI response."

def generate_summary_with_gemini(text: str, max_words: int = 150) -> str:
    """
    Generates a concise summary of the given text using the Gemini API.

    Args:
        text (str): The input text to summarize.
        max_words (int): The maximum number of words for the summary.

    Returns:
        str: The generated summary.
    """
    prompt = f"""
    Summarize the following document content concisely, in no more than {max_words} words.
    Focus on the main points and key information.

    Document Content:
    {text}

    Summary:
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=int(max_words * 1.5)
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating summary with Gemini: {e}")
        return "Could not generate summary."

if __name__ == '__main__':
    print("--- Testing Ask Anything Mode ---")
    dummy_context = [
        "The Amazon rainforest is the largest tropical rainforest in the world, renowned for its biodiversity.",
        "It covers an area of about 5.5 million square kilometers, spanning nine countries.",
        "Deforestation is a major threat to the Amazon, primarily driven by agriculture and logging.",
        "Conservation efforts are crucial to protect its unique ecosystems and indigenous communities."
    ]
    question_1 = "What is the Amazon rainforest known for?"
    answer_1, justification_1 = generate_response_with_gemini(question_1, dummy_context) 
    print(f"\nQuestion: {question_1}")
    print(f"Answer: {answer_1}")
    print(f"Justification: {justification_1}")
    question_2 = "How big is the Amazon rainforest and which countries does it span?"
    answer_2, justification_2 = generate_response_with_gemini(question_2, dummy_context)
    print(f"\nQuestion: {question_2}")
    print(f"Answer: {answer_2}")
    print(f"Justification: {justification_2}")
    question_3 = "What is the capital of France?"
    answer_3, justification_3 = generate_response_with_gemini(question_3, dummy_context)
    print(f"\nQuestion: {question_3}")
    print(f"Answer: {answer_3}")
    print(f"Justification: {justification_3}")
    print("\n--- Testing Summary Generation ---")
    long_text_for_summary = """
    The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device, and the foundational ideas behind AI, were proposed by Alan Turing, who is widely considered the father of AI. The field of AI research was founded at a conference at Dartmouth College in 1956.
    """
    summary = generate_summary_with_gemini(long_text_for_summary, max_words=50)
    print(f"\nOriginal Text:\n{long_text_for_summary}")
    print(f"\nGenerated Summary (approx 50 words):\n{summary}")
