import os
import json
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, send_file, session
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app import db
from app.models.document import Document
from app.models.question import Question, QuestionSet
from app.forms.document_forms import DocumentUploadForm, QuestionGenerationForm
from app.utils.document_processor import process_document
from app.utils.question_generator import generate_questions
from app.utils.pdf_exporter import export_to_pdf
import tempfile
from app.models.course import Course
from datetime import date
import traceback

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
        try:
            file = form.document.data
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                upload_dir = os.path.join(current_app.root_path, 'static/uploads')
                file_path = os.path.join(upload_dir, filename)
                
                # Ensure the upload directory exists
                os.makedirs(upload_dir, exist_ok=True)
                
                # Save the file
                file.save(file_path)
                
                try:
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
                except Exception as e:
                    # Handle text extraction errors
                    if os.path.exists(file_path):
                        os.remove(file_path)  # Clean up the file if processing failed
                    flash(f'Error processing document: {str(e)}', 'danger')
                    print(f"Document processing error: {str(e)}")
            else:
                flash('Invalid file format. Please upload a PDF or Word document.', 'danger')
        except Exception as e:
            flash(f'Error uploading document: {str(e)}', 'danger')
            print(f"Upload error: {str(e)}")
    elif request.method == 'POST':
        # This will show validation errors
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"Error in {getattr(form, field).label.text}: {error}", 'danger')
    
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
        # All questions and question sets will be deleted automatically due to cascade delete
        # in the document's relationships with question_sets and questions
        
        # If the file exists, delete it
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        # Delete the document (cascades to question sets and questions)
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
    
    # Set default date to today if not provided
    if not form.exam_date.data:
        form.exam_date.data = date.today()
    
    if form.validate_on_submit():
        # Generate structured questions for Section A (10 short answer questions)
        section_a_questions = generate_questions(
            document.content,
            10,  # Fixed number of questions for Section A
            'structured',  # Short answer type
            form.difficulty.data
        )
        
        # Generate essay questions for Section B (5 long answer questions)
        section_b_questions = generate_questions(
            document.content,
            5,  # Fixed number of questions for Section B
            'essay',  # Essay type
            form.difficulty.data
        )
        
        # Format the exam date for display
        formatted_exam_date = form.exam_date.data.strftime('%d %B %Y')
        
        # Create a new question set
        question_set = QuestionSet(
            title=f"UBTEB Exam - {document.title} - {date.today().strftime('%Y-%m-%d')}",
            document_id=document.id,
            user_id=current_user.id,
            difficulty=form.difficulty.data,
            exam_series=form.exam_series.data,
            programme_list=form.programme_list.data,
            paper_name=form.paper_name.data,
            paper_code=form.paper_code.data,
            year_semester=form.year_semester.data,
            exam_date=formatted_exam_date
        )
        
        db.session.add(question_set)
        db.session.flush()  # Get the question_set ID before committing
        
        # Save all questions to database with the question_set_id
        for q in section_a_questions:
            # Convert answer to JSON string if it's a dictionary
            answer = q['answer']
            if isinstance(answer, dict):
                answer = json.dumps(answer)
                
            question = Question(
                content=q['question'],
                answer=answer,
                options=None,
                question_type='section_a',  # Mark as Section A
                difficulty=form.difficulty.data,
                document_id=document.id,
                user_id=current_user.id,
                question_set_id=question_set.id
            )
            db.session.add(question)
        
        for q in section_b_questions:
            # Convert answer to JSON string if it's a dictionary
            answer = q['answer']
            if isinstance(answer, dict):
                answer = json.dumps(answer)
                
            question = Question(
                content=q['question'],
                answer=answer,
                options=None,
                question_type='section_b',  # Mark as Section B
                difficulty=form.difficulty.data,
                document_id=document.id,
                user_id=current_user.id,
                question_set_id=question_set.id
            )
            db.session.add(question)
        
        db.session.commit()
        
        total_questions = len(section_a_questions) + len(section_b_questions)
        flash(f'{total_questions} questions generated successfully in new question set!', 'success')
        return redirect(url_for('questions.view_question_set', question_set_id=question_set.id))
    
    return render_template('questions/generate.html', form=form, document=document)

@questions.route('/documents/<int:document_id>/questions')
@login_required
def view_questions(document_id):
    """Redirect to question sets view"""
    return redirect(url_for('questions.list_question_sets', document_id=document_id))

@questions.route('/documents/<int:document_id>/export', methods=['GET'])
@login_required
def export_questions(document_id):
    """Find the most recent question set and export it"""
    document = Document.query.get_or_404(document_id)
    
    # Check if the document belongs to the current user
    if document.user_id != current_user.id:
        flash('You do not have permission to export questions for this document.', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    # Find the most recent question set
    question_set = QuestionSet.query.filter_by(document_id=document.id).order_by(QuestionSet.created_at.desc()).first()
    
    if question_set:
        return redirect(url_for('questions.export_question_set', question_set_id=question_set.id))
    else:
        flash('No question sets found to export.', 'warning')
        return redirect(url_for('questions.list_question_sets', document_id=document_id))

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
        question_set_id = question.question_set_id
        return redirect(url_for('questions.view_question_set', question_set_id=question_set_id))
    
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
    
    question_set_id = question.question_set_id
    
    db.session.delete(question)
    db.session.commit()
    
    flash('Question deleted successfully!', 'success')
    return redirect(url_for('questions.view_question_set', question_set_id=question_set_id))

@questions.route('/documents/<int:document_id>/question-sets')
@login_required
def list_question_sets(document_id):
    """List all question sets for a document"""
    document = Document.query.get_or_404(document_id)
    
    # Check if the document belongs to the current user
    if document.user_id != current_user.id:
        flash('You do not have permission to view question sets for this document.', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    question_sets = QuestionSet.query.filter_by(document_id=document.id).order_by(QuestionSet.created_at.desc()).all()
    
    return render_template('questions/question_sets.html', document=document, question_sets=question_sets)

@questions.route('/question-sets/<int:question_set_id>')
@login_required
def view_question_set(question_set_id):
    """View a specific question set"""
    question_set = QuestionSet.query.get_or_404(question_set_id)
    
    # Check if the question set belongs to the current user
    if question_set.user_id != current_user.id:
        flash('You do not have permission to view this question set.', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    document = Document.query.get_or_404(question_set.document_id)
    questions = Question.query.filter_by(question_set_id=question_set.id).all()
    
    return render_template('questions/view_question_set.html', document=document, question_set=question_set, questions=questions)

@questions.route('/question-sets/<int:question_set_id>/export')
@login_required
def export_question_set(question_set_id):
    """Export a question set to PDF"""
    question_set = QuestionSet.query.get_or_404(question_set_id)
    
    # Check if the question set belongs to the current user
    if question_set.user_id != current_user.id:
        flash('You do not have permission to export this question set.', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    document = Document.query.get_or_404(question_set.document_id)
    questions = Question.query.filter_by(question_set_id=question_set.id).all()
    
    if not questions:
        flash('No questions to export.', 'warning')
        return redirect(url_for('questions.view_question_set', question_set_id=question_set.id))
    
    # Create a temporary file for the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        pdf_path = temp_file.name
    
    # Get exam metadata from the question set
    exam_metadata = {
        'exam_series': question_set.exam_series,
        'programme_list': question_set.programme_list,
        'paper_name': question_set.paper_name,
        'paper_code': question_set.paper_code,
        'year_semester': question_set.year_semester,
        'exam_date': question_set.exam_date
    }
    
    # Generate PDF
    export_to_pdf(document, questions, pdf_path, exam_metadata)
    
    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=f"{question_set.title}.pdf",
        mimetype='application/pdf'
    )

@questions.route('/question-sets/<int:question_set_id>/delete', methods=['POST'])
@login_required
def delete_question_set(question_set_id):
    """Delete a question set"""
    question_set = QuestionSet.query.get_or_404(question_set_id)
    
    # Verify ownership
    if question_set.user_id != current_user.id:
        flash('You do not have permission to delete this question set.', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    document_id = question_set.document_id
    
    try:
        db.session.delete(question_set)  # This will also delete associated questions due to cascade
        db.session.commit()
        flash('Question set deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred: {str(e)}', 'danger')
    
    return redirect(url_for('questions.list_question_sets', document_id=document_id)) 