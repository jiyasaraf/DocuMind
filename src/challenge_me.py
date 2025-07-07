# challenge_me.py
import google.generativeai as genai
import os
from dotenv import load_dotenv
import random
import re
from typing import Tuple

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please set it.")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-2.0-flash')

def generate_challenge_questions(document_text: str, num_questions: int = 3) -> list[str]:
    """
    Generates concise, logic-based or comprehension-focused questions from the document text.

    Args:
        document_text (str): The full text of the document.
        num_questions (int): The number of questions to generate.

    Returns:
        list[str]: A list of generated questions.
    """
    if not document_text:
        return []

    prompt = f"""
    Based on the following document, generate {num_questions} distinct, *concise*, logic-based or comprehension-focused questions.
    The questions should require understanding and inference, not just direct recall.
    Each question should be a single, clear sentence.

    Document:
    {document_text}

    Questions:
    1.
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7, # Higher temperature for more creative questions
                max_output_tokens=250 # Reduced tokens for conciseness
            )
        )
        questions_raw = response.text.strip().split('\n')
        
        parsed_questions = []
        for q in questions_raw:
            # Use a more robust regex to capture the question text
            match = re.match(r'^\s*\d+\.?\s*(.*)', q.strip())
            if match:
                parsed_questions.append(match.group(1).strip())
        
        return parsed_questions[:num_questions]

    except Exception as e:
        print(f"Error generating challenge questions with Gemini: {e}")
        return ["Could not generate challenge questions."]

def evaluate_user_answer(
    question: str,
    user_answer: str,
    document_text: str
) -> Tuple[bool, str, int, str]:
    """
    Evaluates a user's answer to a challenge question based on the document text,
    providing a score, detailed justification, and a desired answer snippet if applicable.

    Args:
        question (str): The challenge question.
        user_answer (str): The user's answer.
        document_text (str): The full text of the document for grounding evaluation.

    Returns:
        Tuple[bool, str, int, str]: A tuple containing:
                                    - is_correct (True/False)
                                    - justification (str)
                                    - score (int, out of 10)
                                    - desired_answer_snippet (str, or "N/A")
    """
    if not user_answer or not user_answer.strip():
        return False, "Please provide an answer to evaluate.", 0, "N/A"

    prompt = f"""
    You are an AI evaluator. Your task is to assess the 'User Answer' to the 'Question' based *only* on the 'Document Content'.

    Provide the following:
    1.  **Evaluation Status:** State 'Correct' or 'Incorrect'.
    2.  **Score:** Assign a score from 0 to 10.
        * 10: Perfect answer, fully accurate, complete, and directly addresses the question based on the document.
        * 7-9: Mostly correct, but may lack minor details, depth, or specific connections to the document, or could be phrased more precisely.
        * 4-6: Partially correct, contains some relevant information but is significantly incomplete, contains inaccuracies, or misses the main point of the question.
        * 0-3: Largely incorrect, irrelevant, nonsensical, or demonstrates a severe misunderstanding of the question or document content.
    3.  **Justification:** Explain clearly and specifically why the answer received its score. Detail what aspects were correct, what was incorrect, and what relevant information was missing. Reference the document content explicitly where applicable to support your evaluation.
    4.  **Desired Answer Snippet (if applicable):** If the user's answer is not perfect (score < 10), provide a concise snippet or synthesis of the *missing or incorrectly addressed information* from the 'Document Content' that would lead to a perfect score. This should complement the user's answer to form a complete correct response. If the answer is perfect, state "N/A".

    Document Content:
    {document_text}

    Question: {question}
    User Answer: {user_answer}

    Evaluation:
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1, # Low temperature for factual evaluation
                max_output_tokens=500 # Increased tokens for more detailed justification and snippet
            )
        )
        evaluation_text = response.text.strip()

        # Initialize default values
        is_correct = False
        score = 0
        justification = "Could not parse justification."
        desired_answer_snippet = "N/A"

        # Parse Evaluation Status
        status_match = re.search(r'\*\*Evaluation Status:\*\*\s*(Correct|Incorrect)', evaluation_text, re.IGNORECASE)
        if status_match:
            is_correct = (status_match.group(1).lower() == 'correct')

        # Parse Score
        score_match = re.search(r'\*\*Score:\*\*\s*(\d+)', evaluation_text) # Simplified regex for score
        if score_match:
            score = int(score_match.group(1))

        # Parse Justification and Desired Answer Snippet
        # Use a regex to find the sections
        justification_match = re.search(r'\*\*Justification:\*\*(.*?)(?=\*\*Desired Answer Snippet:|\Z)', evaluation_text, re.DOTALL)
        if justification_match:
            justification = justification_match.group(1).strip()

        snippet_match = re.search(r'\*\*Desired Answer Snippet:\*\*(.*)', evaluation_text, re.DOTALL)
        if snippet_match:
            desired_answer_snippet = snippet_match.group(1).strip()
            if desired_answer_snippet.lower() == "n/a":
                desired_answer_snippet = "N/A" # Ensure consistent "N/A" for perfect answers

        return is_correct, justification, score, desired_answer_snippet

    except Exception as e:
        print(f"Error evaluating user answer with Gemini: {e}")
        return False, f"An error occurred during evaluation: {e}", 0, "N/A"

if __name__ == '__main__':
    print("--- Testing Challenge Me Mode ---")
    dummy_document_text = """
    The theory of relativity, developed by Albert Einstein, comprises two interrelated theories: special relativity and general relativity. Special relativity applies to all physical phenomena in the absence of gravity. General relativity explains the law of gravitation and its relation to other forces of nature; it uses the concept of spacetime curvature. One of the key implications of special relativity is the mass-energy equivalence formula E=mc².
    """
    print("\n--- Generating Challenge Questions ---")
    questions = generate_challenge_questions(dummy_document_text, num_questions=2)
    for i, q in enumerate(questions):
        print(f"Question {i+1}: {q}")
    if questions:
        test_question = questions[0]
        print(f"\n--- Evaluating User Answer for: '{test_question}' ---\n")
        
        # Test case 1: Correct answer
        correct_answer = "Special relativity deals with physics without gravity, while general relativity explains gravity using spacetime curvature."
        is_correct, justification, score, snippet = evaluate_user_answer(test_question, correct_answer, dummy_document_text)
        print(f"User Answer (Correct): '{correct_answer}'")
        print(f"Evaluation: {'Correct' if is_correct else 'Incorrect'}")
        print(f"Score: {score}/10")
        print(f"Justification: {justification}")
        print(f"Desired Answer Snippet: {snippet}")
        
        print("\n" + "="*50 + "\n")

        # Test case 2: Incorrect answer
        incorrect_answer = "Einstein developed the theory of evolution."
        is_correct, justification, score, snippet = evaluate_user_answer(test_question, incorrect_answer, dummy_document_text)
        print(f"User Answer (Incorrect): '{incorrect_answer}'")
        print(f"Evaluation: {'Correct' if is_correct else 'Incorrect'}")
        print(f"Score: {score}/10")
        print(f"Justification: {justification}")
        print(f"Desired Answer Snippet: {snippet}")

        print("\n" + "="*50 + "\n")

        # Test case 3: Partially correct/incomplete answer
        partial_answer = "Special relativity is about physics without gravity."
        is_correct, justification, score, snippet = evaluate_user_answer(test_question, partial_answer, dummy_document_text)
        print(f"User Answer (Partial): '{partial_answer}'")
        print(f"Evaluation: {'Correct' if is_correct else 'Incorrect'}")
        print(f"Score: {score}/10")
        print(f"Justification: {justification}")
        print(f"Desired Answer Snippet: {snippet}")

        print("\n" + "="*50 + "\n")

        # Test case 4: Empty answer
        empty_answer = ""
        is_correct, justification, score, snippet = evaluate_user_answer(test_question, empty_answer, dummy_document_text)
        print(f"User Answer (Empty): '{empty_answer}'")
        print(f"Evaluation: {'Correct' if is_correct else 'Incorrect'}")
        print(f"Score: {score}/10")
        print(f"Justification: {justification}")
        print(f"Desired Answer Snippet: {snippet}")
