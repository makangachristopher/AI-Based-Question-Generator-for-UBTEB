from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import string

def export_to_pdf(document, questions, output_path):
    """
    Export questions and answers to PDF
    
    Args:
        document (Document): Document object
        questions (list): List of Question objects
        output_path (str): Path to save the PDF
    """
    # Create PDF document
    pdf = SimpleDocTemplate(output_path, pagesize=letter)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=20
    )
    
    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10
    )
    
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=6
    )
    
    # Create content
    content = []
    
    # Add title
    content.append(Paragraph(f"Questions for: {document.title}", title_style))
    content.append(Spacer(1, 20))
    
    # Add questions section
    content.append(Paragraph("Questions", heading_style))
    content.append(Spacer(1, 10))
    
    # Add questions
    for i, question in enumerate(questions):
        # Question number and text
        content.append(Paragraph(f"Q{i+1}. {question.content}", normal_style))
        
        # If multiple choice, add options
        if question.question_type == 'multiple_choice' and question.options:
            for j, option in enumerate(question.options):
                option_letter = string.ascii_uppercase[j]
                content.append(Paragraph(f"    {option_letter}. {option}", normal_style))
        
        content.append(Spacer(1, 10))
    
    # Add answers section
    content.append(Paragraph("Answers", heading_style))
    content.append(Spacer(1, 10))
    
    # Create a table for answers
    answer_data = [["Question", "Answer"]]
    
    for i, question in enumerate(questions):
        if question.question_type == 'multiple_choice' and question.options:
            # Find the index of the correct answer in options
            correct_index = question.options.index(question.answer)
            answer_text = f"{string.ascii_uppercase[correct_index]}. {question.answer}"
        else:
            answer_text = question.answer
        
        answer_data.append([f"Q{i+1}", answer_text])
    
    # Create the table
    answer_table = Table(answer_data, colWidths=[80, 400])
    
    # Style the table
    answer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    content.append(answer_table)
    
    # Build the PDF
    pdf.build(content) 