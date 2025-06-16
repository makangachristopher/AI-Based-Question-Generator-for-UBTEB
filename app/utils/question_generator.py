import re
import random
import os
import json
import socket
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Perplexity AI API client
perplexity_api_key = os.getenv('API')
perplexity_client = None

# Check for internet connection
def has_internet_connection():
    try:
        # Try to connect to a reliable server
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

# Initialize Perplexity if API key exists and internet is available
if perplexity_api_key and has_internet_connection():
    try:
        perplexity_client = OpenAI(
            api_key=perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )
        print("Perplexity AI API initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Perplexity AI API: {str(e)}")
        perplexity_client = None

def generate_questions(text, num_questions=7, question_type='structured', difficulty='medium', use_perplexity=True):
    """
    Generate questions from document text using Perplexity AI API
    
    Args:
        text (str): Document text
        num_questions (int): Number of questions to generate (1-50)
        question_type (str): Type of questions to generate ('structured', 'essay', or 'section_a')
        difficulty (str): Difficulty level ('easy', 'medium', 'hard')
        use_perplexity (bool): Whether to use Perplexity API for generation when available
        
    Returns:
        list: List of generated questions with answers
    """
    # Validate and cap number of questions
    num_questions = min(max(1, int(num_questions)), 50)
    
    # Check internet connection again (in case it changed since initialization)
    internet_available = has_internet_connection()
    
    # If Perplexity should be used and is available with internet connection
    if use_perplexity and perplexity_client and internet_available:
        print("Using Perplexity AI API for question generation...")
        perplexity_questions = generate_questions_with_perplexity(text, num_questions, question_type, difficulty)
        return perplexity_questions
    else:
        # No offline fallback available - return error message
        print("Error: Perplexity AI API is not available and no offline model is configured.")
        return [{
            'question': 'Unable to generate questions - API unavailable',
            'answer': 'Please check your internet connection and API configuration.',
            'context': '',
            'type': question_type,
            'confidence': 0.0
        }]

def generate_questions_with_perplexity(text, num_questions, question_type, difficulty):
    """
    Generate questions using the Perplexity AI API
    """
    questions = []

    # Improved prompt engineering
    if question_type == 'section_a':
        system_prompt = (
            "You are an expert exam question generator. Your task is to create high-quality, concise, and direct Section A exam questions for educational assessments. "
            "Each question must start with an action verb (e.g., Define, State, Explain, Outline, Identify) and be focused on a specific concept from the provided text. "
            "Each question should be followed by a brief, accurate answer based only on the text."
        )
        user_prompt = f"""
Based on the following text, generate exactly {num_questions} Section A style questions of {difficulty} difficulty.
- Each question must start with an action verb (Define, State, Explain, Outline, Identify, etc.).
- Each question should be clear, direct, and focused on a specific concept from the text.
- For each question, provide a brief, accurate answer based only on the text.
- Format your response as a JSON array, where each item has a 'question' and an 'answer' field.

Text: {text}

Example:
[
  {{"question": "Define the term marketing.", "answer": "Marketing is the process of..."}},
  {{"question": "State two characteristics of a market economy.", "answer": "1. Private ownership 2. Freedom of choice"}}
]
"""
    elif question_type == 'essay':
        system_prompt = (
            "You are an expert exam question generator. Your task is to create high-quality, open-ended essay questions that require detailed answers and critical thinking. "
            "Questions should encourage analysis, evaluation, or synthesis, and each should be followed by a brief outline of what a good answer should include, based only on the text."
        )
        user_prompt = f"""
Based on the following text, generate exactly {num_questions} essay questions of {difficulty} difficulty.
- Each question should be open-ended and encourage analysis, evaluation, or synthesis.
- For each question, provide a brief outline of what a good answer should include, based only on the text.
- Format your response as a JSON array, where each item has a 'question' and an 'answer' field.

Text: {text}

Example:
[
  {{"question": "Discuss the impact of technology on modern marketing.", "answer": "A good answer should include: 1. Examples of technology in marketing, 2. Effects on reach and efficiency, 3. Challenges introduced by technology."}}
]
"""
    else:  # structured questions
        system_prompt = (
            "You are an expert exam question generator. Your task is to create high-quality, clear, and focused structured questions that test understanding of specific concepts from the provided text. "
            "Each question should have a definite answer that can be found in or inferred from the text, and each should be followed by a brief, accurate answer."
        )
        user_prompt = f"""
Based on the following text, generate exactly {num_questions} structured questions of {difficulty} difficulty.
- Each question should be clear, focused, and have a definite answer that can be found in or inferred from the text.
- For each question, provide a brief, accurate answer based only on the text.
- Format your response as a JSON array, where each item has a 'question' and an 'answer' field.

Text: {text}

Example:
[
  {{"question": "What is the main function of a database index?", "answer": "To speed up data retrieval operations."}}
]
"""

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            response = perplexity_client.chat.completions.create(
                model="r1-1776",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            content = response.choices[0].message.content
            
            # Try to parse the JSON response
            try:
                if "```json" in content and "```" in content.split("```json")[1]:
                    json_str = content.split("```json")[1].split("```", 1)[0].strip()
                    generated_questions = json.loads(json_str)
                elif "```" in content and "```" in content.split("```", 1)[1]:
                    json_str = content.split("```", 1)[1].split("```", 1)[0].strip()
                    generated_questions = json.loads(json_str)
                else:
                    generated_questions = json.loads(content)
                    
                # Validate output: must be a list of dicts with 'question' and 'answer', and correct length
                if (
                    isinstance(generated_questions, list)
                    and all(isinstance(q, dict) and 'question' in q and 'answer' in q for q in generated_questions)
                    and len(generated_questions) == num_questions
                ):
                    for q in generated_questions:
                        questions.append({
                            'question': q['question'],
                            'answer': q['answer'],
                            'context': text[:200] + '...',
                            'type': question_type,
                            'confidence': 0.95
                        })
                    break  # Success
                else:
                    # If not valid, try again
                    if attempt == max_retries:
                        print("Perplexity output invalid or incomplete after all retries.")
                        return generate_fallback_questions(text, num_questions, question_type, difficulty)
                    continue
                    
            except Exception as e:
                if attempt == max_retries:
                    print(f"Perplexity output parsing failed: {str(e)}.")
                    return generate_fallback_questions(text, num_questions, question_type, difficulty)
                continue
                
        except Exception as e:
            if attempt == max_retries:
                print(f"Error using Perplexity AI API: {str(e)}.")
                return generate_fallback_questions(text, num_questions, question_type, difficulty)
            continue

    # Ensure we have the requested number of questions
    if len(questions) > num_questions:
        questions = questions[:num_questions]
    elif len(questions) < num_questions:
        # If we still don't have enough questions, generate simple fallback questions
        additional_questions = generate_fallback_questions(text, num_questions - len(questions), question_type, difficulty)
        questions.extend(additional_questions)
        
    return questions

def generate_fallback_questions(text, num_questions, question_type, difficulty):
    """
    Generate simple fallback questions when API is unavailable
    """
    questions = []
    
    # Split text into sentences for basic processing
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    
    if not sentences:
        sentences = [text[:200]]  # Use first 200 chars if no good sentences
    
    # Shuffle sentences to get variety
    random.shuffle(sentences)
    
    for i in range(min(num_questions, len(sentences))):
        sentence = sentences[i]
        
        # Extract some words for question generation
        words = sentence.split()
        important_words = [w for w in words if len(w) > 4 and w.lower() not in ['this', 'that', 'with', 'from', 'they', 'have', 'been', 'will', 'were', 'would', 'could', 'should']]
        
        if important_words:
            subject = important_words[0].strip('.,;:?!')
        else:
            subject = "the topic"
        
        # Generate questions based on type
        if question_type == 'section_a':
            templates = [
                f"Define {subject}.",
                f"State two characteristics of {subject}.",
                f"Explain the concept of {subject}.",
                f"Outline the importance of {subject}.",
                f"Identify key features of {subject}."
            ]
        elif question_type == 'essay':
            templates = [
                f"Discuss the significance of {subject} in detail.",
                f"Analyze the role of {subject} and its implications.",
                f"Evaluate the importance of {subject} with examples.",
                f"Examine the relationship between {subject} and related concepts.",
                f"Critically assess the impact of {subject}."
            ]
        else:  # structured
            templates = [
                f"What is {subject}?",
                f"How does {subject} work?",
                f"Why is {subject} important?",
                f"What are the main features of {subject}?",
                f"What role does {subject} play?"
            ]
        
        question_text = random.choice(templates)
        answer = f"Based on the text: {sentence[:100]}..."
        
        questions.append({
            'question': question_text,
            'answer': answer,
            'context': sentence,
            'type': question_type,
            'confidence': 0.6
        })
    
    return questions