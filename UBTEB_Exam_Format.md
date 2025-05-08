# UBTEB Exam Format Implementation

## Overview

The system has been updated to generate questions in the Uganda Business and Technical Examinations Board (UBTEB) format. This format structures exams into two sections:

1. **Section A**: Short answer questions worth 20 marks total (2 marks each)
2. **Section B**: Essay questions worth 80 marks total (20 marks each, answer any FOUR)

## Changes Made

### Question Generation

- Removed options for selecting number of questions and question type
- All exams now follow a standardized format:
  - 10 questions in Section A (structured/short answer format) - 2 marks each
  - 5 questions in Section B (essay format) - 20 marks each (answer any FOUR)
- Only the difficulty level can be selected (Easy, Medium, Hard, or Mixed)

### Mark Distribution

- Total exam marks: 100
- Section A: 20 marks (10 questions × 2 marks each)
- Section B: 80 marks (4 questions × 20 marks each)
- Candidates must answer all questions in Section A
- Candidates choose any FOUR questions from the five provided in Section B

### Question Types

- Questions are now categorized as:
  - `section_a`: Short answer questions (2 marks each)
  - `section_b`: Essay questions requiring detailed responses (20 marks each)

### Templates and Views

- Updated templates to display questions organized by sections
- Modified the generation form to show information about the UBTEB format
- Updated the question view to show separate sections with appropriate instructions

### PDF Export

- PDF export now follows the UBTEB exam format
- Includes proper exam instructions and sections
- Shows mark allocation for each question
- Formatted for official exam presentation

## Working with the New Format

### Generating Questions

1. Upload a document or select an existing one
2. Select the difficulty level
3. Click "Generate Questions"
4. The system will automatically create:
   - 10 Section A questions (2 marks each)
   - 5 Section B essay questions (20 marks each)

### Editing Questions

Questions can still be edited individually to refine their content.

### Exporting to PDF

The exported PDF will follow the official UBTEB format with:
- Proper header and title
- Instructions to candidates
- Section A with all questions (20 marks total)
- Section B with essay questions (80 marks total) 