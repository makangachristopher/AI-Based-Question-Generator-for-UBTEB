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

def generate_questions(text, num_questions=7, question_type='structured', difficulty='medium'):
    """
    Generate questions from document text
    
    Args:
        text (str): Document text
        num_questions (int): Number of questions to generate (1-50)
        question_type (str): Type of questions to generate ('structured', 'essay', or 'section_a')
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
    extra_factor = 3
    selected_sentences = [s[0] for s in sentence_scores[:num_questions * extra_factor]]
    random.shuffle(selected_sentences)
    
    # Generate questions based on the type
    if question_type == 'essay':
        return ensure_question_count(
            generate_essay_questions(selected_sentences, num_questions, difficulty),
            num_questions,
            question_type,
            selected_sentences,
            difficulty
        )
    elif question_type == 'section_a':
        return ensure_question_count(
            generate_section_a_questions(selected_sentences, num_questions, difficulty),
            num_questions,
            question_type,
            selected_sentences,
            difficulty
        )
    else:  # Default to structured questions
        return ensure_question_count(
            generate_structured_questions(selected_sentences, num_questions, difficulty),
            num_questions,
            'structured',
            selected_sentences,
            difficulty
        )

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
        remaining_sentences = sentences
    
    # Generate additional questions using the remaining sentences
    if question_type == 'essay':
        additional_questions = generate_essay_questions(
            remaining_sentences, 
            additional_needed, 
            difficulty
        )
    else:
        additional_questions = generate_structured_questions(
            remaining_sentences, 
            additional_needed, 
            difficulty
        )
    
    # If we still don't have enough, create simple questions from remaining text
    still_needed = target_count - (len(questions) + len(additional_questions))
    if still_needed > 0:
        for i in range(still_needed):
            # Pick a sentence we haven't used or the most information-rich one
            sentence_scores = [(s, len(s.split())) for s in sentences 
                             if s not in [q['context'] for q in questions + additional_questions]]
            if not sentence_scores:
                sentence_scores = [(s, len(s.split())) for s in sentences]
            
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            context = sentence_scores[0][0] if sentence_scores else random.choice(sentences)
            
            # Create a section A style question
            # Extract key subjects from the context
            words = context.split()
            nouns = []
            
            for i in range(0, len(words) - 1):
                # Simple heuristic: words not in common stop words might be subjects
                if words[i].lower() not in ['the', 'a', 'an', 'of', 'for', 'and', 'or', 'but', 'in', 'on', 'at']:
                    nouns.append(words[i])
            
            # Select a subject for the question
            if nouns:
                subject = random.choice(nouns).strip('.,;:?!')
                # If the subject is too short or empty after stripping, try to get a phrase
                if not subject or len(subject) < 4:
                    # Find the original subject before stripping
                    original_subject = random.choice(nouns)
                    # Try to find the original noun's position
                    try:
                        idx = words.index(original_subject)
                        if idx < len(words) - 2:
                            subject = ' '.join(words[idx:idx+3]).strip('.,;:?!')
                        else:
                            subject = original_subject  # Fallback to original if we can't get a phrase
                    except ValueError:
                        # If we can't find the original subject for some reason, use fallback
                        subject = "marketing"
            else:
                # Fallback if no suitable nouns found
                subject = "marketing"
            
            # Ensure subject is never empty
            if not subject or subject.strip() == "":
                subject = "marketing"
            
            # Section A style templates
            templates = [
                f"Define the term {subject}.",
                f"State two characteristics of {subject}.",
                f"Explain the concept of {subject}.",
                f"Outline two features of {subject}.",
                f"Identify two examples of {subject}."
            ]
            
            question_text = random.choice(templates)
            answer = f"Key points about {subject} from the text."
            
            additional_questions.append({
                'question': question_text,
                'answer': answer,
                'context': context,
                'type': question_type,
                'confidence': 0.7
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

def generate_section_a_questions(sentences, num_questions, difficulty):
    """
    Generate Section A style questions that are direct and concise
    
    Args:
        sentences (list): List of sentences to generate questions from
        num_questions (int): Number of questions to generate
        difficulty (str): Difficulty level ('easy', 'medium', 'hard')
        
    Returns:
        list: List of generated Section A style questions with answers
    """
    questions = []
    confidence_base = {'easy': 0.90, 'medium': 0.85, 'hard': 0.80, 'mixed': 0.85}
    
    # Group sentences into paragraphs to get more context
    paragraphs = []
    current_paragraph = []
    
    for sentence in sentences:
        current_paragraph.append(sentence)
        if len(current_paragraph) >= 2:  # Consider 2+ sentences as a paragraph for section A
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
    
    # Add any remaining sentences as a paragraph
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    # If we don't have enough paragraphs, repeat some
    while len(paragraphs) < num_questions:
        paragraphs.extend(paragraphs[:num_questions - len(paragraphs)])
    
    # Shuffle and select paragraphs
    random.shuffle(paragraphs)
    selected_paragraphs = paragraphs[:num_questions]
    
    # Section A question templates based on difficulty
    section_a_templates = {
        'easy': [
            "Define the term {}.",
            "State two characteristics of {}.",
            "Identify two examples of {}.",
            "List two types of {}.",
            "Mention two benefits of {}."
        ],
        'medium': [
            "Explain the concept of {}.",
            "Describe how {} works.",
            "Outline two features of {}.",
            "Distinguish between {} and related concepts.",
            "Summarize the importance of {}."
        ],
        'hard': [
            "Analyze the role of {} in the given context.",
            "Compare and contrast two aspects of {}.",
            "Evaluate the significance of {}.",
            "Examine the relationship between {} and other factors.",
            "Discuss the implications of {}."
        ]
    }
    
    # Use mixed difficulty if specified
    current_templates = section_a_templates.get(
        difficulty if difficulty != 'mixed' else random.choice(['easy', 'medium', 'hard'])
    )
    
    for idx, paragraph in enumerate(selected_paragraphs):
        if idx >= num_questions:
            break
            
        # Extract key subjects from the paragraph
        words = paragraph.split()
        nouns = []
        
        for i in range(0, len(words) - 1):
            # Simple heuristic: words not in common stop words might be subjects
            if words[i].lower() not in ['the', 'a', 'an', 'of', 'for', 'and', 'or', 'but', 'in', 'on', 'at']:
                nouns.append(words[i])
        
        # Select a subject for the question
        if nouns:
            subject = random.choice(nouns).strip('.,;:?!')
            # If the subject is too short or empty after stripping, try to get a phrase
            if not subject or len(subject) < 4:
                # Find the original subject before stripping
                original_subject = random.choice(nouns)
                # Try to find the original noun's position
                try:
                    idx = words.index(original_subject)
                    if idx < len(words) - 2:
                        subject = ' '.join(words[idx:idx+3]).strip('.,;:?!')
                    else:
                        subject = original_subject  # Fallback to original if we can't get a phrase
                except ValueError:
                    # If we can't find the original subject for some reason, use fallback
                    subject = "marketing"
        else:
            # Fallback if no suitable nouns found
            subject = "marketing"
        
        # Ensure subject is never empty
        if not subject or subject.strip() == "":
            subject = "marketing"
        
        # Generate the question using a template
        template = random.choice(current_templates)
        question_text = template.format(subject)
        
        # Generate a concise answer
        answer = f"Key points about {subject} from the text."
        
        questions.append({
            'question': question_text,
            'answer': answer,
            'context': paragraph,
            'type': 'section_a',
            'confidence': confidence_base[difficulty if difficulty != 'mixed' else 'medium'] * (0.9 + 0.2 * random.random())
        })
    
    return questions

def generate_essay_questions(sentences, num_questions, difficulty):
    """
    Generate essay questions that require detailed answers
    
    Args:
        sentences (list): List of sentences to generate questions from
        num_questions (int): Number of questions to generate
        difficulty (str): Difficulty level ('easy', 'medium', 'hard')
        
    Returns:
        list: List of generated essay questions with sample answers
    """
    questions = []
    confidence_base = {'easy': 0.85, 'medium': 0.75, 'hard': 0.65, 'mixed': 0.75}
    
    # Group sentences into paragraphs to get more context
    paragraphs = []
    current_paragraph = []
    
    for sentence in sentences:
        current_paragraph.append(sentence)
        if len(current_paragraph) >= 3:  # Consider 3+ sentences as a paragraph
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
    
    # Add any remaining sentences as a paragraph
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    # If we don't have enough paragraphs, repeat some
    while len(paragraphs) < num_questions:
        paragraphs.extend(paragraphs[:num_questions - len(paragraphs)])
    
    # Shuffle and select paragraphs
    random.shuffle(paragraphs)
    selected_paragraphs = paragraphs[:num_questions]
    
    # Essay-type question prompts based on difficulty
    essay_prompts = {
        'easy': [
            "Explain in detail {}.",
            "Discuss the importance of {} in the given context.",
            "Describe the main concepts related to {}.",
            "Elaborate on the relationship between {} and the main topic.",
            "Write a comprehensive explanation of {}."
        ],
        'medium': [
            "Analyze the implications of {} and discuss its significance.",
            "Compare and contrast different aspects of {} and evaluate their importance.",
            "Critically examine the role of {} in the broader context.",
            "Evaluate the effectiveness of {} and provide examples to support your answer.",
            "Investigate the factors that influence {} and explain their impact."
        ],
        'hard': [
            "Critically evaluate the arguments for and against {} and develop your own position.",
            "Synthesize information about {} and formulate a comprehensive theory or framework.",
            "Assess the validity of different approaches to understanding {} and propose improvements.",
            "Formulate a detailed argument about the significance of {} with reference to theoretical perspectives.",
            "Develop a critical analysis of {} and its implications for future developments in this field."
        ]
    }
    
    # Use mixed difficulty if specified
    current_prompts = essay_prompts.get(
        difficulty if difficulty != 'mixed' else random.choice(['easy', 'medium', 'hard'])
    )
    
    for idx, paragraph in enumerate(selected_paragraphs):
        if idx >= num_questions:
            break
            
        # Extract key subjects from the paragraph
        words = paragraph.split()
        nouns = []
        
        for i in range(0, len(words) - 1):
            # Simple heuristic: words not in common stop words might be subjects
            if words[i].lower() not in ['the', 'a', 'an', 'of', 'for', 'and', 'or', 'but', 'in', 'on', 'at']:
                nouns.append(words[i])
        
        # Select a subject for the question
        if nouns:
            subject = random.choice(nouns).strip('.,;:?!')
            # If the subject is too short or empty after stripping, try to get a phrase
            if not subject or len(subject) < 4:
                # Find the original subject before stripping
                original_subject = random.choice(nouns)
                # Try to find the original noun's position
                try:
                    idx = words.index(original_subject)
                    if idx < len(words) - 2:
                        subject = ' '.join(words[idx:idx+3]).strip('.,;:?!')
                    else:
                        subject = original_subject  # Fallback to original if we can't get a phrase
                except ValueError:
                    # If we can't find the original subject for some reason, use fallback
                    subject = "this topic"
        else:
            # Fallback if no suitable nouns found
            subject = "this topic"
        
        # Ensure subject is never empty
        if not subject or subject.strip() == "":
            subject = "this topic"
        
        # Generate the question using a prompt template
        prompt = random.choice(current_prompts)
        question_text = prompt.format(subject)
        
        # Clean up the question text
        question_text = clean_question_text(question_text)
        
        # Generate a sample answer framework (this would be just guidance)
        sample_answer = f"A comprehensive answer about {subject} should include:\n\n"
        sample_answer += f"1. Introduction to {subject} and its context\n"
        sample_answer += f"2. Key aspects and characteristics of {subject}\n"
        sample_answer += f"3. Analysis of the importance and implications of {subject}\n"
        sample_answer += f"4. Examples and evidence related to {subject}\n"
        sample_answer += f"5. Conclusion summarizing the main points about {subject}"
        
        questions.append({
            'question': question_text,
            'answer': sample_answer,
            'context': paragraph,
            'confidence': confidence_base[difficulty if difficulty != 'mixed' else 'medium'] * (0.9 + 0.2 * random.random())
        })
    
    return questions