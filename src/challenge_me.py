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
    Generates logic-based or comprehension-focused questions from the document text.
    This is a preliminary implementation and can be significantly enhanced.

    Args:
        document_text (str): The full text of the document.
        num_questions (int): The number of questions to generate.

    Returns:
        list[str]: A list of generated questions.
    """
    if not document_text:
        return []

    prompt = f"""
    Based on the following document, generate {num_questions} distinct, logic-based or comprehension-focused questions.
    The questions should require understanding and inference, not just direct recall.
    Present each question clearly.

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
                max_output_tokens=500
            )
        )
        questions_raw = response.text.strip().split('\n')
        
        parsed_questions = []
        for q in questions_raw:
            match = re.match(r'^\d+\.?\s*(.*)', q.strip())
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
) -> Tuple[bool, str]:
    """
    Evaluates a user's answer to a challenge question based on the document text.
    This is a preliminary implementation.

    Args:
        question (str): The challenge question.
        user_answer (str): The user's answer.
        document_text (str): The full text of the document for grounding evaluation.

    Returns:
        Tuple[bool, str]: A tuple indicating if the answer is correct (True/False)
                          and a justification for the evaluation.
    """
    if not user_answer:
        return False, "Please provide an answer."

    prompt = f"""
    You are an AI evaluator. Your task is to assess if the 'User Answer' correctly and logically answers the 'Question',
    based *only* on the 'Document Content'. Provide a clear justification for your evaluation.
    State if the answer is 'Correct' or 'Incorrect'.

    Document Content:
    {document_text}

    Question: {question}
    User Answer: {user_answer}

    Evaluation:
    Is the User Answer correct based on the Document Content? (Correct/Incorrect):
    Justification:
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1, # Low temperature for factual evaluation
                max_output_tokens=300
            )
        )
        evaluation_text = response.text.strip()

        is_correct = "Correct" in evaluation_text
        justification_match = re.search(r'Justification:\s*(.*)', evaluation_text, re.DOTALL)
        justification = justification_match.group(1).strip() if justification_match else "No specific justification provided."

        return is_correct, justification

    except Exception as e:
        print(f"Error evaluating user answer with Gemini: {e}")
        return False, "An error occurred during evaluation."

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
        print(f"\n--- Evaluating User Answer for: '{test_question}' ---")
        correct_answer = "Special relativity deals with physics without gravity, while general relativity explains gravity using spacetime curvature."
        is_correct, justification = evaluate_user_answer(test_question, correct_answer, dummy_document_text)
        print(f"User Answer (Correct): '{correct_answer}'")
        print(f"Evaluation: {'Correct' if is_correct else 'Incorrect'}")
        print(f"Justification: {justification}")
        incorrect_answer = "Einstein developed the theory of evolution."
        is_correct, justification = evaluate_user_answer(test_question, incorrect_answer, dummy_document_text)
        print(f"\nUser Answer (Incorrect): '{incorrect_answer}'")
        print(f"Evaluation: {'Correct' if is_correct else 'Incorrect'}")
        print(f"Justification: {justification}")
