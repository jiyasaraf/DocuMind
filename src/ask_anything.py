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

    # Combine context chunks into a single string
    # Add an index to each chunk for justification
    formatted_context = "\n\n".join([f"Context [{i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])

    # Construct the prompt for Gemini
    # Emphasize grounding the answer in the provided context and providing justification.
    prompt = f"""
    You are an AI assistant designed to answer questions based *only* on the provided document context.
    If the question cannot be answered from the context, state that you cannot answer it.
    Always justify your answer by referencing the specific context chunk(s) (e.g., "This is supported by Context [1] and Context [3]").

    Document Context:
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
        # Extract the text from the response
        answer_text = response.text.strip()

        # Simple heuristic to extract justification.
        # This can be made more robust if Gemini provides structured output for justification.
        justification_match = re.search(r'(This is supported by Context \[.*?\](?: and Context \[.*?\])*)', answer_text, re.IGNORECASE)
        justification = ""
        if justification_match:
            justification = justification_match.group(0)
            # Remove justification from the main answer text
            answer_text = answer_text.replace(justification, "").strip()
            # Clean up any trailing punctuation if justification was at the end
            if answer_text.endswith('.'):
                answer_text = answer_text[:-1].strip()

        return answer_text, justification

    except Exception as e:
        print(f"Error generating content with Gemini: {e}")
        return "An error occurred while trying to generate a response. Please try again.", ""

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
