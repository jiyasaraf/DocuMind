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
    2.
    3.
    ...
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=300)
        )
        questions_raw = response.text.strip()
        # Parse questions, handling various numbering/bullet formats
        questions = re.findall(r"^\s*\d+\.\s*(.*)", questions_raw, re.MULTILINE)
        if not questions:
            # Fallback for other formats like bullet points or just lines
            questions = [q.strip() for q in questions_raw.split('\n') if q.strip() and not q.strip().startswith(('Answer', 'Justification'))]
        
        return questions[:num_questions] # Ensure only requested number of questions are returned
    except Exception as e:
        print(f"Error generating challenge questions with Gemini: {e}")
        return []

def evaluate_user_answer(
    question: str,
    user_answer: str,
    document_text: str,
    max_output_tokens: int = 500
) -> Tuple[bool, str, int, str]:
    """
    Evaluates a user's answer against the document context using the Gemini API.

    Args:
        question (str): The question asked.
        user_answer (str): The user's provided answer.
        document_text (str): The full text of the document for grounding.
        max_output_tokens (int): Maximum number of tokens for the generated evaluation.

    Returns:
        Tuple[bool, str, int, str]: A tuple containing:
                                     - is_correct (bool): True if the answer is largely correct.
                                     - justification (str): Explanation for the evaluation and score.
                                     - score (int): A score from 0-10.
                                     - desired_answer_snippet (str): A snippet from the document
                                                                     containing the correct answer, or "N/A".
    """
    if not document_text:
        return False, "Evaluation cannot be performed: document context is missing.", 0, "N/A"
    if not user_answer.strip():
        return False, "No answer provided by the user.", 0, "N/A"

    prompt = f"""
    You are an AI assistant tasked with evaluating a user's answer to a question based on a provided document.
    Your evaluation should be fair, comprehensive, and grounded strictly in the document content.

    **Evaluation Criteria:**
    1.  **Accuracy & Relevance (60%)**: How accurate is the user's answer based *only* on the document? Is it directly relevant to the question and the document's content? Irrelevant or hallucinated answers should receive a very low score (0-2).
    2.  **Completeness (20%)**: Does the user's answer cover all key aspects of the question as presented in the document?
    3.  **Clarity & Conciseness (10%)**: Is the answer clear, easy to understand, and to the point?
    4.  **Minimal Hallucination (10%)**: Does the answer avoid making up information not present in the document?

    **Instructions:**
    -   First, determine the correct answer to the question based *only* on the `Document` provided.
    -   Then, compare the `User Answer` to the correct answer derived from the `Document`.
    -   Provide an `Evaluation Status` (Correct, Incorrect, Partially Correct).
    -   Assign a `Score` from 0 to 10 based on the criteria. A score of 0 should be given for completely irrelevant, nonsensical, or hallucinated answers.
    -   Provide a detailed `Justification` for the score, explaining why the user's answer was correct, incorrect, or partially correct, citing specifics from the document.
    -   Provide a `Desired Answer Snippet` from the document that best answers the question. If the document does not contain the answer, state "N/A".

    **Question:** {question}

    **User Answer:** {user_answer}

    **Document:**
    {document_text}

    ---
    **Evaluation Format:**
    Evaluation Status: [Correct/Incorrect/Partially Correct]
    Score: [0-10]/10
    Justification: [Detailed explanation based on criteria and document]
    Desired Answer Snippet (if applicable): [Relevant snippet from document or "N/A"]
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=max_output_tokens)
        )
        response_text = response.text.strip()

        # Extract information using regex
        status_match = re.search(r"Evaluation Status: (.*)", response_text)
        score_match = re.search(r"Score: (\d+)/10", response_text)
        justification_match = re.search(r"Justification: (.*?)(?=\nDesired Answer Snippet:|$)", response_text, re.DOTALL)
        snippet_match = re.search(r"Desired Answer Snippet \(if applicable\): (.*)", response_text, re.DOTALL)

        # Stricter interpretation of 'is_correct'
        status_text = status_match.group(1).lower() if status_match else ""
        score = int(score_match.group(1)) if score_match else 0
        
        # An answer is considered "correct" only if the status explicitly says so AND the score is reasonably high (e.g., >= 7)
        is_correct = ("correct" in status_text and score >= 7) 

        justification = justification_match.group(1).strip() if justification_match else "No justification provided."
        desired_answer_snippet = snippet_match.group(1).strip() if snippet_match else "N/A"

        return is_correct, justification, score, desired_answer_snippet

    except Exception as e:
        print(f"Error evaluating user answer with Gemini: {e}")
        return False, "An error occurred during evaluation.", 0, "N/A"

if __name__ == '__main__':
    dummy_document_text = """
    Special relativity is a theory of space and time. It was proposed by Albert Einstein in 1905.
    It deals with the relationship between space and time, and how they are affected by motion.
    Key concepts include the constancy of the speed of light for all inertial observers and the relativity of simultaneity.
    It does not incorporate gravity, which is addressed by general relativity.
    The theory has profound implications for mass-energy equivalence, famously expressed as E=mc².
    """

    test_question = "What are the key concepts of special relativity and what does it not incorporate?"

    print("--- Testing Answer Evaluation ---")

    # Test case 1: Correct answer
    correct_answer = "The key concepts are the constancy of the speed of light and the relativity of simultaneity. It does not incorporate gravity."
    is_correct, justification, score, snippet = evaluate_user_answer(test_question, correct_answer, dummy_document_text)
    print(f"User Answer (Correct): '{correct_answer}'")
    print(f"Evaluation: {'Correct' if is_correct else 'Incorrect'}")
    print(f"Score: {score}/10")
    print(f"Justification: {justification}")
    print(f"Desired Answer Snippet: {snippet}")

    print("\n" + "="*50 + "\n")

    # Test case 2: Incorrect answer
    incorrect_answer = "Special relativity is all about quantum mechanics and the theory of everything solution."
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

    print("\n" + "="*50 + "\n")
    
    # Test case 5: Vague/Irrelevant answer (new test case)
    vague_answer = "hethrre y345"
    is_correct, justification, score, snippet = evaluate_user_answer(test_question, vague_answer, dummy_document_text)
    print(f"User Answer (Vague): '{vague_answer}'")
    print(f"Evaluation: {'Correct' if is_correct else 'Incorrect'}")
    print(f"Score: {score}/10")
    print(f"Justification: {justification}")
    print(f"Desired Answer Snippet: {snippet}")

    print("\n" + "="*50 + "\n")

    # Test case 6: Generate challenge questions
    print("--- Testing Question Generation ---")
    generated_questions = generate_challenge_questions(dummy_document_text, num_questions=2)
    print("Generated Questions:")
    for i, q in enumerate(generated_questions):
        print(f"{i+1}. {q}")
