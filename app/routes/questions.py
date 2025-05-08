import os
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, send_file
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app import db
from app.models.document import Document
from app.models.question import Question
from app.forms.document_forms import DocumentUploadForm, QuestionGenerationForm
from app.utils.document_processor import process_document
from app.utils.question_generator import generate_questions
from app.utils.pdf_exporter import export_to_pdf
import tempfile
from app.models.course import Course

questions = Blueprint('questions', __name__)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@questions.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_document():
    """Document upload route"""
    form = DocumentUploadForm()
    
    # Populate the course choices
    user_courses = Course.query.filter_by(user_id=current_user.id).all()
    form.course_id.choices = [(0, 'No Course')] + [(course.id, course.title) for course in user_courses]
    
    if form.validate_on_submit():
        file = form.document.data
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.root_path, 'static/uploads', filename)
            
            # Ensure the upload directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            file.save(file_path)
            
            # Extract text from document
            document_text = process_document(file_path)
            
            # Save document to database
            document = Document(
                title=form.title.data,
                filename=filename,
                file_path=file_path,
                content=document_text,
                user_id=current_user.id
            )
            
            # Set course_id if a course was selected
            if form.course_id.data != 0:
                document.course_id = form.course_id.data
            
            db.session.add(document)
            db.session.commit()
            
            flash('Document uploaded successfully!', 'success')
            return redirect(url_for('questions.generate_questions_form', document_id=document.id))
        else:
            flash('Invalid file format. Please upload a PDF or Word document.', 'danger')
    
    return render_template('questions/upload.html', form=form)

@questions.route('/documents')
@login_required
def list_documents():
    """List user's documents"""
    documents = Document.query.filter_by(user_id=current_user.id).all()
    return render_template('questions/documents.html', documents=documents)

@questions.route('/documents/<int:document_id>')
@login_required
def view_document(document_id):
    """View document details"""
    document = Document.query.get_or_404(document_id)
    
    # Check if the document belongs to the current user
    if document.user_id != current_user.id:
        flash('You do not have permission to view this document.', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    return render_template('questions/view_document.html', document=document)

@questions.route('/document/<int:document_id>/delete', methods=['POST'])
@login_required
def delete_document(document_id):
    document = Document.query.get_or_404(document_id)
    
    # Verify ownership
    if document.user_id != current_user.id:
        flash('You do not have permission to delete this document.', 'error')
        return redirect(url_for('questions.list_documents'))
    
    try:
        # Delete associated questions first
        Question.query.filter_by(document_id=document.id).delete()
        
        # Delete the document
        db.session.delete(document)
        db.session.commit()
        
        flash('Document and associated questions have been deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while deleting the document: {str(e)}', 'error')
    
    return redirect(url_for('questions.list_documents'))

@questions.route('/documents/<int:document_id>/generate', methods=['GET', 'POST'])
@login_required
def generate_questions_form(document_id):
    """Question generation form"""
    document = Document.query.get_or_404(document_id)
    
    # Check if the document belongs to the current user
    if document.user_id != current_user.id:
        flash('You do not have permission to generate questions for this document.', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    form = QuestionGenerationForm()
    
    if form.validate_on_submit():
        # Generate questions
        generated_questions = generate_questions(
            document.content,
            form.num_questions.data,
            form.question_type.data,
            form.difficulty.data
        )
        
        # Save questions to database
        for q in generated_questions:
            question = Question(
                content=q['question'],
                answer=q['answer'],
                options=q.get('options', None),  # Only for multiple choice
                question_type=form.question_type.data,
                difficulty=form.difficulty.data,
                document_id=document.id,
                user_id=current_user.id
            )
            db.session.add(question)
        
        db.session.commit()
        
        flash(f'{len(generated_questions)} questions generated successfully!', 'success')
        return redirect(url_for('questions.view_questions', document_id=document.id))
    
    return render_template('questions/generate.html', form=form, document=document)

@questions.route('/documents/<int:document_id>/questions')
@login_required
def view_questions(document_id):
    """View generated questions"""
    document = Document.query.get_or_404(document_id)
    
    # Check if the document belongs to the current user
    if document.user_id != current_user.id:
        flash('You do not have permission to view questions for this document.', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    questions = Question.query.filter_by(document_id=document.id).all()
    
    return render_template('questions/view_questions.html', document=document, questions=questions)

@questions.route('/questions/<int:question_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_question(question_id):
    """Edit question"""
    question = Question.query.get_or_404(question_id)
    
    # Check if the question belongs to the current user
    if question.user_id != current_user.id:
        flash('You do not have permission to edit this question.', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    if request.method == 'POST':
        question.content = request.form.get('content')
        question.answer = request.form.get('answer')
        
        if question.question_type == 'multiple_choice':
            options = []
            for i in range(4):  # Assuming 4 options for multiple choice
                option = request.form.get(f'option_{i}')
                if option:
                    options.append(option)
            question.options = options
        
        db.session.commit()
        
        flash('Question updated successfully!', 'success')
        return redirect(url_for('questions.view_questions', document_id=question.document_id))
    
    return render_template('questions/edit_question.html', question=question)

@questions.route('/questions/<int:question_id>/delete', methods=['POST'])
@login_required
def delete_question(question_id):
    """Delete question"""
    question = Question.query.get_or_404(question_id)
    
    # Check if the question belongs to the current user
    if question.user_id != current_user.id:
        flash('You do not have permission to delete this question.', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    document_id = question.document_id
    
    db.session.delete(question)
    db.session.commit()
    
    flash('Question deleted successfully!', 'success')
    return redirect(url_for('questions.view_questions', document_id=document_id))

@questions.route('/documents/<int:document_id>/export', methods=['GET'])
@login_required
def export_questions(document_id):
    """Export questions to PDF"""
    document = Document.query.get_or_404(document_id)
    
    # Check if the document belongs to the current user
    if document.user_id != current_user.id:
        flash('You do not have permission to export questions for this document.', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    questions = Question.query.filter_by(document_id=document.id).all()
    
    if not questions:
        flash('No questions to export.', 'warning')
        return redirect(url_for('questions.view_questions', document_id=document.id))
    
    # Create a temporary file for the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        pdf_path = temp_file.name
    
    # Generate PDF
    export_to_pdf(document, questions, pdf_path)
    
    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=f"{document.title}_questions.pdf",
        mimetype='application/pdf'
    ) 