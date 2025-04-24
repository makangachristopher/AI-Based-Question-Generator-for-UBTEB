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
    
    # Select more sentences than needed to ensure we can reach the requested question count
    # We'll select twice as many sentences as questions to have sufficient material
    extra_factor = 3
    selected_sentences = [s[0] for s in sentence_scores[:num_questions * extra_factor]]
    random.shuffle(selected_sentences)
    
    # Generate questions based on the type
    if question_type == 'multiple_choice':
        return ensure_question_count(
            generate_multiple_choice_questions(selected_sentences, num_questions, difficulty),
            num_questions,
            question_type,
            selected_sentences,
            difficulty
        )
    elif question_type == 'structured':
        return ensure_question_count(
            generate_structured_questions(selected_sentences, num_questions, difficulty),
            num_questions,
            question_type,
            selected_sentences,
            difficulty
        )
    else:  # 'both'
        mc_count = num_questions // 2
        structured_count = num_questions - mc_count
        
        mc_questions = generate_multiple_choice_questions(
            selected_sentences[:len(selected_sentences) // 2], 
            mc_count, 
            difficulty
        )
        
        structured_questions = generate_structured_questions(
            selected_sentences[len(selected_sentences) // 2:], 
            structured_count, 
            difficulty
        )
        
        # Ensure we have the exact count requested for each type
        mc_questions = ensure_question_count(
            mc_questions,
            mc_count,
            'multiple_choice',
            selected_sentences,
            difficulty
        )
        
        structured_questions = ensure_question_count(
            structured_questions,
            structured_count,
            'structured',
            selected_sentences,
            difficulty
        )
        
        return mc_questions + structured_questions

def ensure_question_count(questions, target_count, question_type, sentences, difficulty):
    """
    Ensure we have exactly the requested number of questions.
    If we have too few, generate more. If we have too many, trim.
    
    Args:
        questions (list): Currently generated questions
        target_count (int): Number of questions we need
        question_type (str): Type of questions to generate
        sentences (list): Available sentences to generate from
        difficulty (str): Difficulty level
        
    Returns:
        list: List with exactly target_count questions
    """
    if len(questions) == target_count:
        return questions
        
    # If we have too many questions, trim to the target count
    if len(questions) > target_count:
        # Sort by confidence score to keep the best questions
        questions.sort(key=lambda q: q['confidence'], reverse=True)
        return questions[:target_count]
        
    # If we have too few questions, we need to generate more
    additional_needed = target_count - len(questions)
    
    # First, try to use any remaining sentences we haven't tried yet
    used_contexts = [q['context'] for q in questions]
    remaining_sentences = [s for s in sentences if s not in used_contexts]
    
    # If we have no remaining sentences, we'll reuse some, preferring ones not already used
    if not remaining_sentences:
        # Reuse sentences, prioritizing those we haven't used yet
        remaining_sentences = sentences
        
    # Retry generation with remaining sentences
    additional_questions = []
    if question_type == 'multiple_choice':
        additional_questions = generate_multiple_choice_questions(
            remaining_sentences, 
            additional_needed, 
            difficulty
        )
    else:  # structured
        additional_questions = generate_structured_questions(
            remaining_sentences, 
            additional_needed, 
            difficulty
        )
    
    # If we still don't have enough, create context-based questions instead of generic ones
    still_needed = target_count - (len(questions) + len(additional_questions))
    if still_needed > 0:
        for i in range(still_needed):
            # Pick a sentence we haven't used or the most information-rich one
            sentence_scores = [(s, len(s.split())) for s in sentences if s not in [q['context'] for q in questions + additional_questions]]
            if not sentence_scores:
                sentence_scores = [(s, len(s.split())) for s in sentences]
                
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            if sentence_scores:
                context = sentence_scores[0][0]
            else:
                context = random.choice(sentences)
                
            # Extract key information from the context
            words = context.split()
            if len(words) >= 5:
                subject_idx = random.randint(0, len(words) // 3)
                subject = words[subject_idx]
                
                if question_type == 'multiple_choice':
                    question_text = f"What does the text say about {subject}?"
                    options = [
                        " ".join(words[subject_idx:subject_idx+3]),
                        " ".join(words[subject_idx:subject_idx+2]),
                        " ".join(words[max(0, subject_idx-2):subject_idx]),
                        " ".join(words[min(len(words)-3, subject_idx+3):min(len(words), subject_idx+6)])
                    ]
                    random.shuffle(options)
                    answer = options[0]
                    
                    additional_questions.append({
                        'question': question_text,
                        'answer': answer,
                        'options': options,
                        'context': context,
                        'type': 'multiple_choice',
                        'confidence': 0.8
                    })
                else:
                    # Create a factual question directly from the text
                    if len(words) > 10:
                        answer_idx = random.randint(len(words)//2, min(len(words)-1, len(words)//2 + 5))
                        answer = words[answer_idx]
                        question_text = f"According to the text, what comes after '{' '.join(words[max(0, answer_idx-3):answer_idx])}'?"
                    else:
                        question_text = f"What information does the text provide about {subject}?"
                        answer = context
                        
                    additional_questions.append({
                        'question': question_text,
                        'answer': answer,
                        'context': context,
                        'type': 'structured',
                        'confidence': 0.8
                    })
    
    # Combine original questions with additional ones
    return questions + additional_questions[:additional_needed]

def generate_structured_questions(sentences, num_questions, difficulty):
    """Generate structured questions from sentences"""
    questions = []
    num_to_generate = min(num_questions, len(sentences))
    
    # Increase the number of attempts per sentence to ensure we get more questions
    max_attempts_per_sentence = 4
    max_sentences_to_try = min(len(sentences), num_questions * 2)
    
    # Try generating from each sentence until we have enough questions
    for i in range(max_sentences_to_try):
        if i >= len(sentences):
            break
            
        if len(questions) >= num_questions:
            break
            
        context = sentences[i]
        question_text = None
        answer = None
        
        # Modified prompt to strongly enforce staying within the content (anti-hallucination)
        prompt = f"Generate exactly one question from this SPECIFIC text. The question MUST be answerable ONLY from the text provided. Do not make up information or add external knowledge: {context}"
        
        # Try multiple times per sentence to get a valid question
        for attempt in range(max_attempts_per_sentence):
            generated_output = question_generator(
                prompt, 
                max_length=64, 
                num_return_sequences=1,
                do_sample=True,
                temperature=0.5,  # Lower temperature to reduce creativity/hallucination
                top_p=0.85,
                no_repeat_ngram_size=3,
                early_stopping=True  # Enable early stopping
            )[0]['generated_text']
            
            # Clean and validate the question
            cleaned_question = clean_question_text(generated_output)
            if cleaned_question and '?' in cleaned_question:
                # Verify the answer exists in the context with higher threshold
                potential_answer = answer_generator(
                    question=cleaned_question,
                    context=context
                )
                
                # Stricter confidence threshold to ensure only relevant questions
                confidence_threshold = 0.6
                
                if potential_answer['score'] > confidence_threshold:
                    # Verify answer appears literally in the context
                    if potential_answer['answer'] in context:
                        question_text = cleaned_question
                        answer = potential_answer
                        break  # Got a good question, exit retry loop
        
        # Only add new questions if they're different from existing ones
        if question_text and answer:
            # Check if this question is too similar to ones we already have
            is_duplicate = False
            for existing_q in questions:
                # Basic similarity check - if questions share many words
                existing_words = set(existing_q['question'].lower().split())
                new_words = set(question_text.lower().split())
                overlap = len(existing_words.intersection(new_words)) / len(existing_words.union(new_words))
                
                if overlap > 0.7:  # More than 70% word overlap
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                questions.append({
                    'question': question_text,
                    'answer': answer['answer'],
                    'context': context,
                    'type': 'structured',
                    'confidence': answer['score']
                })
    
    # Return the number of questions requested, or as many as we could generate
    return questions[:num_questions]

def generate_multiple_choice_questions(sentences, num_questions, difficulty):
    """Generate multiple-choice questions from sentences"""
    questions = []
    
    # Increase the number of attempts per sentence to ensure we get more questions
    max_attempts_per_sentence = 4
    max_sentences_to_try = min(len(sentences), num_questions * 2)
    
    # Try generating from each sentence until we have enough questions
    for i in range(max_sentences_to_try):
        if i >= len(sentences):
            break
            
        if len(questions) >= num_questions:
            break
            
        context = sentences[i]
        
        # Modified prompt to strictly enforce relevance to the context (anti-hallucination)
        prompt = f"Based ONLY on this SPECIFIC text, generate one multiple choice question with 4 options. The question and correct answer MUST be factually contained within the text. Do not make up any information: {context}"
        
        # Try multiple times per sentence to get a valid question
        for attempt in range(max_attempts_per_sentence):
            question_text = question_generator(
                prompt, 
                max_length=64, 
                num_return_sequences=1,
                do_sample=True,
                temperature=0.5,  # Lower temperature to reduce creativity/hallucination
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
                
                # Stricter confidence threshold to ensure only relevant questions
                confidence_threshold = 0.6
                
                if answer_result['score'] > confidence_threshold:
                    # Verify answer appears literally in the context
                    if answer_result['answer'] in context:
                        question_text = cleaned_question
                        correct_answer = answer_result['answer']
                        
                        # Check if this question is too similar to ones we already have
                        is_duplicate = False
                        for existing_q in questions:
                            # Basic similarity check - if questions share many words
                            existing_words = set(existing_q['question'].lower().split())
                            new_words = set(question_text.lower().split())
                            overlap = len(existing_words.intersection(new_words)) / len(existing_words.union(new_words))
                            
                            if overlap > 0.7:  # More than 70% word overlap
                                is_duplicate = True
                                break
                                
                        if not is_duplicate:
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
                            
                            # Break out of the retry loop since we got a good question
                            break
    
    # Return the number of questions requested, or as many as we could generate
    return questions[:num_questions]

def generate_distractors(context, correct_answer, difficulty):
    """Generate wrong options for multiple choice questions"""
    # Adjust temperature based on difficulty
    temp = 0.5  # Lower temperature to reduce hallucination
    if difficulty == 'easy':
        temp = 0.4
    elif difficulty == 'hard':
        temp = 0.6
    
    # Modified prompt to enforce context-based distractors and prevent hallucination
    prompt = f"""Based ONLY on this text: {context}
Generate 3 plausible but incorrect answer options that are closely related to the context.
Correct answer: {correct_answer}
The options MUST be mentioned in or derived from the text and be different from: {correct_answer}
Do not make up any information that's not in the text."""
    
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
                # Only add distractor if it's somehow related to the context
                if any(word in context.lower() for word in clean_line.lower().split() if len(word) > 3):
                    distractors.append(clean_line)
    
    # If we didn't get enough distractors, generate some based on text extraction
    while len(distractors) < 3:
        # Extract noun phrases or entities from context
        words = context.split()
        if len(words) > 5:
            # Try to extract a phrase from the context that's not the answer
            start_idx = random.randint(0, len(words) - 3)
            phrase_length = random.randint(1, 3)
            phrase = " ".join(words[start_idx:start_idx + phrase_length])
            
            if phrase and phrase != correct_answer and phrase not in distractors:
                distractors.append(phrase)
            else:
                # Fall back to a word-based distractor that appears in the context
                context_words = [word for word in words if len(word) > 3 and word.lower() not in correct_answer.lower()]
                if context_words:
                    distractor = random.choice(context_words)
                    if distractor not in distractors:
                        distractors.append(distractor)
                else:
                    # Last resort
                    generic = f"Alternative from text {len(distractors) + 1}"
                    distractors.append(generic)
    
    return distractors[:3]

def clean_question_text(text):
    """Clean up generated question text to ensure only one question is returned"""
    # First, split on <sep> token if present and take the first part
    if '<sep>' in text:
        text = text.split('<sep>')[0].strip()
    
    # Remove common prefixes and formatting that might indicate multiple items
    text = re.sub(r'^(Q:|Question:|A:|Answer:|\\d+[\\.\\)]|\\-|\\*)\\s*', '', text, flags=re.IGNORECASE).strip()
    
    # Split by newline, periods followed by space (if not ending the string), or question marks (if not ending)
    # Prioritize splitting by newline as it often separates distinct generated questions
    potential_questions = re.split(r'\\n+|(?<!\\w)\\.\\s+(?=\\w)|\\?(?!$)', text)
    
    # Find the first valid-looking question
    question = None
    for pq in potential_questions:
        pq = pq.strip()
        if len(pq.split()) >= 3 and '?' in pq: # Check length and presence of question mark
            question = pq
            break
        elif len(pq.split()) >= 3 and not '?' in pq: # Check if it's just missing a question mark
            question = pq + '?'
            break
    
    if not question:
        # Fallback if splitting didn't find a clear question
        # Take the original text if it looks plausible, otherwise return None
        if len(text.split()) >= 3:
            question = text
        else:
            return None
    
    # Further cleanup on the selected question
    # Remove any remaining leading numbering/bullets
    question = re.sub(r'^[\\d\\-\\.\\)\\•\\*]+\\s*', '', question).strip()
    
    # Ensure it ends with a single question mark
    question = question.rstrip('. ')
    if not question.endswith('?'):
        question += '?'
    
    # Handle cases where multiple question marks might still exist
    if question.count('?') > 1:
        question = question.split('?')[0] + '?'
    
    # Final length check
    if len(question.split()) < 3:
        return None
    
    return question