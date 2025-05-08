import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem, BulletDrawer

def export_to_pdf(document, questions, output_path):
    """
    Export questions to PDF format following UBTEB exam template
    
    Args:
        document (Document): Document object
        questions (list): List of Question objects
        output_path (str): Path to save the PDF
    """
    # Create a PDF document
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Create custom styles with unique names to avoid conflicts
    custom_title_style = ParagraphStyle(
        name='UbtebTitle',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=1  # Center alignment
    )
    
    custom_subtitle_style = ParagraphStyle(
        name='UbtebSubtitle',
        parent=styles['Heading2'],
        fontSize=12,
        alignment=1  # Center alignment
    )
    
    custom_section_header = ParagraphStyle(
        name='UbtebSectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=10,
        spaceAfter=6
    )
    
    custom_question_style = ParagraphStyle(
        name='UbtebQuestionNumber',
        parent=styles['Normal'],
        fontSize=12,
        fontName='Helvetica-Bold'
    )
    
    # Content elements for the PDF
    elements = []
    
    # Add title
    title = Paragraph("UGANDA BUSINESS AND TECHNICAL EXAMINATIONS BOARD", custom_title_style)
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    # Get course title if available
    course_title = document.course.title if document.course else document.title
    subtitle = Paragraph(f"{course_title.upper()} EXAMINATION", custom_subtitle_style)
    elements.append(subtitle)
    elements.append(Spacer(1, 12))
    
    # Add instructions
    instructions = Paragraph("INSTRUCTIONS TO CANDIDATES:", styles['Normal'])
    elements.append(instructions)
    elements.append(Spacer(1, 6))
    
    # Add exam instructions as bullet points properly
    instruction_items = [
        Paragraph("Attempt ALL questions in Section A (20 Marks).", styles['Normal']),
        Paragraph("Attempt any FOUR questions from Section B (80 Marks).", styles['Normal']),
        Paragraph("Begin each answer on a fresh page.", styles['Normal']),
        Paragraph("Non-programmable calculators may be used.", styles['Normal']),
        Paragraph("Write your name and candidate number on each page of your answer booklet.", styles['Normal'])
    ]
    
    # Create a ListFlowable for the instructions (correct way to handle bullet points)
    bullet_list = ListFlowable(
        instruction_items,
        bulletType='bullet',
        leftIndent=20,
        bulletFontName='Helvetica',
        bulletFontSize=10
    )
    elements.append(bullet_list)
    
    elements.append(Spacer(1, 20))
    
    # Separate questions by type
    section_a_questions = [q for q in questions if q.question_type == 'section_a']
    section_b_questions = [q for q in questions if q.question_type == 'section_b']
    
    # Add Section A
    if section_a_questions:
        section_a_header = Paragraph("SECTION A: ANSWER ALL QUESTIONS (20 Marks)", custom_section_header)
        elements.append(section_a_header)
        elements.append(Spacer(1, 10))
        
        for i, question in enumerate(section_a_questions):
            # Add question number and marks (2 marks per Section A question)
            question_num = Paragraph(f"{i+1}. {question.content} (2 Marks)", custom_question_style)
            elements.append(question_num)
            elements.append(Spacer(1, 10))
            
            # Optional: Add space for answer in the exam paper
            elements.append(Spacer(1, 20))
    
    # Add Section B
    if section_b_questions:
        elements.append(Spacer(1, 20))
        section_b_header = Paragraph("SECTION B: ANSWER ANY FOUR QUESTIONS (80 Marks)", custom_section_header)
        elements.append(section_b_header)
        elements.append(Spacer(1, 10))
        
        for i, question in enumerate(section_b_questions):
            # Add question number and marks (20 marks per Section B question)
            question_num = Paragraph(f"{i+1}. {question.content} (20 Marks)", custom_question_style)
            elements.append(question_num)
            elements.append(Spacer(1, 10))
            
            # Optional: Add space for answer in the exam paper
            elements.append(Spacer(1, 40))
    
    # Build the PDF
    doc.build(elements)
    
    return output_path 