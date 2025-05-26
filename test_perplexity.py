import sys
import os
import json
from app.utils.question_generator import generate_questions

def main():
    # Check if a text file is provided as an argument
    if len(sys.argv) < 2:
        print("Usage: python test_perplexity.py <text_file> [num_questions] [question_type] [difficulty]")
        print("Example: python test_perplexity.py sample_text.txt 5 section_a medium")
        return
    
    # Get parameters
    text_file = sys.argv[1]
    num_questions = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    question_type = sys.argv[3] if len(sys.argv) > 3 else 'section_a'
    difficulty = sys.argv[4] if len(sys.argv) > 4 else 'medium'
    
    # Validate question_type
    valid_types = ['structured', 'essay', 'section_a']
    if question_type not in valid_types:
        print(f"Invalid question type. Must be one of: {', '.join(valid_types)}")
        return
    
    # Validate difficulty
    valid_difficulties = ['easy', 'medium', 'hard', 'mixed']
    if difficulty not in valid_difficulties:
        print(f"Invalid difficulty. Must be one of: {', '.join(valid_difficulties)}")
        return
    
    # Read the text file
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return
    
    print(f"Generating {num_questions} {question_type} questions with {difficulty} difficulty using Perplexity AI...")
    
    # Generate questions using Perplexity
    questions = generate_questions(
        text=text,
        num_questions=num_questions,
        question_type=question_type,
        difficulty=difficulty,
        use_perplexity=True
    )
    
    # Print the generated questions
    print(f"\nGenerated {len(questions)} questions:\n")
    for i, q in enumerate(questions, 1):
        print(f"Question {i}: {q['question']}")
        print(f"Answer: {q['answer']}")
        print("-" * 50)
    
    # Save questions to a JSON file
    output_file = f"perplexity_{question_type}_questions.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=4)
    
    print(f"\nQuestions saved to {output_file}")

if __name__ == "__main__":
    main()
