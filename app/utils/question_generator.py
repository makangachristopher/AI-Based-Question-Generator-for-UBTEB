import torch
import re
import random
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, pipeline

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(
    'valhalla/t5-base-e2e-qg',
    model_max_length=512
)
model = AutoModelForSeq2SeqLM.from_pretrained('valhalla/t5-base-e2e-qg')

# Check for CUDA availability
device = 0 if torch.cuda.is_available() else -1

# Initialize the pipelines
question_generator = pipeline(
    'text2text-generation',
    model=model,
    tokenizer=tokenizer,
    device=device
)
answer_generator = pipeline('question-answering', model='deepset/roberta-base-squad2', device=device)

def generate_questions(text, num_questions=5, question_type='both', difficulty='medium'):
    """
    Generate questions from document text
    
    Args:
        text (str): Document text
        num_questions (int): Number of questions to generate
        question_type (str): Type of questions to generate ('multiple_choice', 'structured', or 'both')
        difficulty (str): Difficulty level ('easy', 'medium', 'hard')
        
    Returns:
        list: List of generated questions with answers
    """
    # Validate and cap number of questions
    num_questions = min(max(1, int(num_questions)), 50)
    
    # Preprocess the text
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 5]
    
    if len(sentences) < 3:
        return []
    
    # Use TF-IDF to identify important sentences
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    importance_scores = tfidf_matrix.sum(axis=1).A1
    
    # Create a list of (sentence, score) tuples and sort by score
    sentence_scores = list(zip(sentences, importance_scores))
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select only the number of sentences we need
    selected_sentences = [s[0] for s in sentence_scores[:num_questions]]
    random.shuffle(selected_sentences)
    
    # Generate questions based on the type
    if question_type == 'multiple_choice':
        return generate_multiple_choice_questions(selected_sentences, num_questions, difficulty)
    elif question_type == 'structured':
        return generate_structured_questions(selected_sentences, num_questions, difficulty)
    else:  # 'both'
        mc_count = num_questions // 2
        structured_count = num_questions - mc_count
        
        mc_questions = generate_multiple_choice_questions(
            selected_sentences[:mc_count], 
            mc_count, 
            difficulty
        )
        structured_questions = generate_structured_questions(
            selected_sentences[mc_count:], 
            structured_count, 
            difficulty
        )
        return mc_questions + structured_questions
def generate_structured_questions(sentences, num_questions, difficulty):
    """Generate structured questions from sentences"""
    questions = []
    num_to_generate = min(num_questions, len(sentences))
    
    for i in range(num_to_generate):
        if i >= len(sentences):
            break
            
        context = sentences[i]
        
        # Modified prompt to be more explicit about single question generation
        prompt = f"Generate exactly one question from this text. Do not generate multiple questions or use separators: {context}"
        
        # Try up to 3 times to get a clean single question
        for attempt in range(3):
            question_text = question_generator(
                prompt, 
                max_length=64, 
                num_return_sequences=1,
                do_sample=True,
                temperature=0.6,
                top_p=0.85,
                no_repeat_ngram_size=3,
                early_stopping=True  # Add early stopping to prevent multiple questions
            )[0]['generated_text']
            
            # Clean and validate the question
            cleaned_question = clean_question_text(question_text)
            if cleaned_question and '?' in cleaned_question:
                # Verify the answer exists in the context
                answer = answer_generator(
                    question=cleaned_question, 
                    context=context
                )
                
                # Only accept questions where the answer confidence is high
                if answer['score'] > 0.7:
                    question_text = cleaned_question
                    break
        
        # Only proceed if we have a valid question with a confident answer
        if question_text and '?' in question_text:
            questions.append({
                'question': question_text,
                'answer': answer['answer'],
                'context': context,
                'type': 'structured',
                'confidence': answer['score']
            })
        
        if len(questions) >= num_questions:
            break
    
    return questions[:num_questions]

def generate_multiple_choice_questions(sentences, num_questions, difficulty):
    """Generate multiple-choice questions from sentences"""
    questions = []
    num_to_generate = min(num_questions, len(sentences))
    
    for i in range(num_to_generate):
        if i >= len(sentences):
            break
            
        context = sentences[i]
        
        # Modified prompt to ensure relevance
        prompt = f"Based on this specific text, generate one multiple choice question that can be answered directly from the text: {context}"
        
        # Try up to 3 times to get a relevant question
        for attempt in range(3):
            question_text = question_generator(
                prompt, 
                max_length=64, 
                num_return_sequences=1,
                do_sample=True,
                temperature=0.6,  # Reduced temperature
                top_p=0.85,
                no_repeat_ngram_size=3
            )[0]['generated_text']
            
            # Clean and validate the question
            cleaned_question = clean_question_text(question_text)
            if cleaned_question and '?' in cleaned_question:
                # Verify the answer exists in the context
                answer_result = answer_generator(
                    question=cleaned_question, 
                    context=context
                )
                
                # Only accept questions where the answer confidence is high
                if answer_result['score'] > 0.7:
                    question_text = cleaned_question
                    correct_answer = answer_result['answer']
                    break
        
        # Only proceed if we have a valid question with a confident answer
        if question_text and '?' in question_text and 'correct_answer' in locals():
            # Generate distractors based on the context
            distractors = generate_distractors(context, correct_answer, difficulty)
            
            # Combine correct answer and distractors
            options = [correct_answer] + distractors[:3]
            random.shuffle(options)
            
            questions.append({
                'question': question_text,
                'answer': correct_answer,
                'options': options,
                'context': context,
                'type': 'multiple_choice',
                'confidence': answer_result['score']
            })
        
        if len(questions) >= num_questions:
            break
    
    return questions[:num_questions]

def generate_distractors(context, correct_answer, difficulty):
    """Generate wrong options for multiple choice questions"""
    # Adjust temperature based on difficulty
    temp = 0.6  # Reduced default temperature
    if difficulty == 'easy':
        temp = 0.5
    elif difficulty == 'hard':
        temp = 0.7
    
    # Modified prompt to ensure relevant distractors
    prompt = f"""Based on this text: {context}
Generate 3 plausible but incorrect answer options that are related to the context.
Correct answer: {correct_answer}
The options should be different from: {correct_answer}"""
    
    results = question_generator(
        prompt,
        max_length=128,
        num_return_sequences=1,
        do_sample=True,
        temperature=temp,
        top_p=0.85
    )[0]['generated_text']
    
    # Parse and clean distractors
    distractors = []
    for line in results.split('\n'):
        line = line.strip()
        if line and line != correct_answer and not line.startswith(("Context:", "Correct answer:", "Generate")):
            # Remove any numbering or bullet points
            clean_line = re.sub(r'^[\d\-\.\)\•\*]+\s*', '', line)
            if clean_line and clean_line not in distractors and clean_line != correct_answer:
                distractors.append(clean_line)
    
    # If we didn't get enough distractors, generate some based on the context
    while len(distractors) < 3:
        # Use answer_generator to find other entities in the context
        probe_question = f"What is another {correct_answer.split()[0]} mentioned in the text?"
        probe_answer = answer_generator(question=probe_question, context=context)
        if probe_answer['answer'] and probe_answer['answer'] != correct_answer:
            distractors.append(probe_answer['answer'])
        else:
            generic = f"Alternative option {len(distractors) + 1}"
            distractors.append(generic)
    
    return distractors[:3]

def clean_question_text(text):
    """Clean up generated question text to ensure only one question is returned"""
    # First, split on <sep> token and take only the first part
    if '<sep>' in text:
        text = text.split('<sep>')[0]
    
    # Remove common prefixes and formatting
    text = re.sub(r'^(Q:|Question:|A:|Answer:|\d+[\.\)]|\-|\*)\s*', '', text, flags=re.IGNORECASE)
    
    # Split on common question separators (newlines, numbers, etc.)
    questions = re.split(r'(?:\d+[\.\)])|(?:\n+)|(?:Question:)', text)
    
    # Clean up and take only the first question
    questions = [q.strip() for q in questions if q.strip()]
    if not questions:
        return None
        
    # Take the first question
    question = questions[0]
    
    # Remove any remaining numbering or bullets
    question = re.sub(r'^[\d\-\.\)\•\*]+\s*', '', question)
    
    # Clean up whitespace
    question = ' '.join(question.split())
    
    # Validate the question
    if len(question.split()) < 3:  # Too short
        return None
        
    if not question.endswith('?'):  # Add question mark if missing
        question += '?'
        
    if question.count('?') > 1:  # Multiple questions
        # Take only up to the first question mark
        question = question.split('?')[0] + '?'
    
    return question