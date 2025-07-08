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

    context_str = "\n".join(context_chunks)

    prompt = f"""
    You are a helpful assistant. Answer the following question ONLY based on the provided context.
    If the answer cannot be found in the context, respond with "I cannot answer this question based on the provided document."
    Always provide a justification for your answer by citing the relevant part of the context.

    Question: {question}

    Context:
    {context_str}

    Answer: [Your answer here]
    Justification: [Cite the relevant part of the context here]
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=max_output_tokens)
        )
        response_text = response.text

        # Use regex to extract answer and justification
        # Make the regex more robust to handle potential variations in spacing/newlines
        answer_match = re.search(r"Answer:\s*(.*?)(?=\nJustification:|$)", response_text, re.DOTALL | re.IGNORECASE)
        justification_match = re.search(r"Justification:\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)

        answer = answer_match.group(1).strip() if answer_match else ""
        justification = justification_match.group(1).strip() if justification_match else ""

        # Fallback: If answer is empty but justification exists, try to use the full response text as answer
        if not answer and justification:
             # If the model didn't explicitly format an "Answer:", use the whole response as the answer
             # and try to extract justification from it.
            answer = response_text.replace(f"Justification: {justification}", "").strip()
            if answer.lower().startswith("answer:"):
                answer = answer[len("answer:"):].strip()
            elif answer.lower().startswith("smart assistant:"): # Handle cases where Streamlit prepends this
                answer = answer[len("smart assistant:"):].strip()
            
        # Final fallback if both are still empty or unparsed
        if not answer and not justification:
             return response_text.strip(), "Could not parse specific justification from response."


        return answer, justification

    except Exception as e:
        print(f"Error generating response with Gemini: {e}")
        return "I encountered an error while trying to answer your question.", ""

def generate_summary_with_gemini(text: str, max_words: int = 150) -> str:
    """
    Generates a concise summary of the provided text using the Gemini API.

    Args:
        text (str): The text to summarize.
        max_words (int): The maximum number of words for the summary.

    Returns:
        str: The generated summary. Returns an empty string if generation fails.
    """
    if not text.strip():
        return "No text provided to summarize."

    prompt = f"""
    Summarize the following document concisely, in no more than {max_words} words.
    Focus on the main points and key information.

    Document:
    {text}

    Summary:
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=max_words * 2) # Allow more tokens for summary generation
        )
        summary = response.text.strip()
        # Optional: Truncate to exact word count if needed, though LLM should largely adhere
        words = summary.split()
        if len(words) > max_words:
            summary = " ".join(words[:max_words]) + "..."
        return summary
    except Exception as e:
        print(f"Error generating summary with Gemini: {e}")
        return "Could not generate a summary due to an error."

if __name__ == '__main__':
    # Dummy context for testing
    dummy_context = [
        "Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals.",
        "Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
        "The term 'artificial intelligence' was coined by John McCarthy in 1956 at the Dartmouth Conference, which is considered the birth of AI as an academic field.",
        "AI research has been defined as the field of study of 'intelligent agents', which refers to any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
        "Machine learning is a subset of AI that focuses on the development of algorithms that allow computers to learn from and make predictions or decisions on data.",
        "Deep learning is a subset of machine learning that uses neural networks with many layers (deep neural networks) to analyze various factors of data."
    ]

    print("--- Testing Question Answering ---")
    question_1 = "What is Artificial Intelligence?"
    answer_1, justification_1 = generate_response_with_gemini(question_1, dummy_context)
    print(f"Question: {question_1}")
    print(f"Answer: {answer_1}")
    print(f"Justification: {justification_1}")

    question_2 = "When was the term 'artificial intelligence' coined and by whom?"
    answer_2, justification_2 = generate_response_with_gemini(question_2, dummy_context)
    print(f"\nQuestion: {question_2}")
    print(f"Answer: {answer_2}")
    print(f"Justification: {justification_2}")

    question_3 = "What is the capital of France?" # Irrelevant question
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
    print(f"Summary (max 50 words):\n{summary}")

    short_text_for_summary = "This is a short sentence."
    summary_short = generate_summary_with_gemini(short_text_for_summary, max_words=10)
    print(f"\nOriginal Text:\n{short_text_for_summary}")
    print(f"Summary (max 10 words):\n{summary_short}")

    empty_text_for_summary = ""
    summary_empty = generate_summary_with_gemini(empty_text_for_summary)
    print(f"\nOriginal Text:\n'{empty_text_for_summary}'")
    print(f"Summary (empty text):\n{summary_empty}")
