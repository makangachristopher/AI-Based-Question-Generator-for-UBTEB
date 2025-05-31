from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_required, current_user
from app import db
from app.models.program import Program
from app.models.course import Course
from app.models.paper import Paper
from app.forms.program_forms import ProgramForm, CourseForm, PaperForm

programs = Blueprint('programs', __name__)

@programs.route('/programs')
@login_required
def list_programs():
    """List user's programs"""
    user_programs = Program.query.filter_by(user_id=current_user.id).all()
    return render_template('programs/programs.html', programs=user_programs)

@programs.route('/programs/new', methods=['GET', 'POST'])
@login_required
def create_program():
    """Create a new program"""
    form = ProgramForm()
    
    if form.validate_on_submit():
        program = Program(
            name=form.name.data,
            description=form.description.data,
            user_id=current_user.id
        )
        
        db.session.add(program)
        db.session.commit()
        
        flash('Program created successfully!', 'success')
        return redirect(url_for('programs.list_programs'))
    
    return render_template('programs/program_form.html', form=form, title='New Program')

@programs.route('/programs/<int:program_id>')
@login_required
def view_program(program_id):
    """View program details and associated courses"""
    program = Program.query.get_or_404(program_id)
    
    # Check if the program belongs to the current user
    if program.user_id != current_user.id:
        flash('You do not have permission to view this program.', 'danger')
        return redirect(url_for('programs.list_programs'))
    
    # Get all courses for this program
    courses = Course.query.filter_by(program_id=program_id).all()
    
    return render_template('programs/view_program.html', program=program, courses=courses)

@programs.route('/programs/<int:program_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_program(program_id):
    """Edit program details"""
    program = Program.query.get_or_404(program_id)
    
    # Check if the program belongs to the current user
    if program.user_id != current_user.id:
        flash('You do not have permission to edit this program.', 'danger')
        return redirect(url_for('programs.list_programs'))
    
    form = ProgramForm()
    
    if form.validate_on_submit():
        program.name = form.name.data
        program.description = form.description.data
        
        db.session.commit()
        
        flash('Program updated successfully!', 'success')
        return redirect(url_for('programs.view_program', program_id=program.id))
    
    elif request.method == 'GET':
        form.name.data = program.name
        form.description.data = program.description
    
    return render_template('programs/program_form.html', form=form, title='Edit Program')

@programs.route('/programs/<int:program_id>/delete', methods=['POST'])
@login_required
def delete_program(program_id):
    """Delete a program and all its courses, papers, and documents"""
    program = Program.query.get_or_404(program_id)
    
    # Check if the program belongs to the current user
    if program.user_id != current_user.id:
        flash('You do not have permission to delete this program.', 'danger')
        return redirect(url_for('programs.list_programs'))
    
    try:
        # Due to cascade delete, all associated courses, papers, and documents will be deleted
        db.session.delete(program)
        db.session.commit()
        flash('Program and all associated items have been deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while deleting the program: {str(e)}', 'error')
    
    return redirect(url_for('programs.list_programs'))

# Course routes within programs
@programs.route('/programs/<int:program_id>/courses/new', methods=['GET', 'POST'])
@login_required
def create_course(program_id):
    """Create a new course within a program"""
    program = Program.query.get_or_404(program_id)
    
    # Check if the program belongs to the current user
    if program.user_id != current_user.id:
        flash('You do not have permission to add courses to this program.', 'danger')
        return redirect(url_for('programs.list_programs'))
    
    form = CourseForm()
    form.program_id.choices = [(program.id, program.name)]
    form.program_id.data = program.id
    
    if form.validate_on_submit():
        course = Course(
            name=form.name.data,
            description=form.description.data,
            program_id=form.program_id.data,
            user_id=current_user.id
        )
        
        db.session.add(course)
        db.session.commit()
        
        flash('Course created successfully!', 'success')
        return redirect(url_for('programs.view_program', program_id=program.id))
    
    return render_template('programs/course_form.html', form=form, program=program, title='New Course')

@programs.route('/courses/<int:course_id>')
@login_required
def view_course(course_id):
    """View course details and associated papers"""
    course = Course.query.get_or_404(course_id)
    
    # Check if the course belongs to the current user
    if course.user_id != current_user.id:
        flash('You do not have permission to view this course.', 'danger')
        return redirect(url_for('programs.list_programs'))
    
    # Get all papers for this course
    papers = Paper.query.filter_by(course_id=course_id).all()
    
    return render_template('programs/view_course.html', course=course, papers=papers)

@programs.route('/courses/<int:course_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_course(course_id):
    """Edit course details"""
    course = Course.query.get_or_404(course_id)
    
    # Check if the course belongs to the current user
    if course.user_id != current_user.id:
        flash('You do not have permission to edit this course.', 'danger')
        return redirect(url_for('programs.list_programs'))
    
    form = CourseForm()
    
    # Get all programs for the current user
    user_programs = Program.query.filter_by(user_id=current_user.id).all()
    form.program_id.choices = [(p.id, p.name) for p in user_programs]
    
    if form.validate_on_submit():
        course.name = form.name.data
        course.description = form.description.data
        course.program_id = form.program_id.data
        
        db.session.commit()
        
        flash('Course updated successfully!', 'success')
        return redirect(url_for('programs.view_course', course_id=course.id))
    
    elif request.method == 'GET':
        form.name.data = course.name
        form.description.data = course.description
        form.program_id.data = course.program_id
    
    return render_template('programs/course_form.html', form=form, title='Edit Course')

@programs.route('/courses/<int:course_id>/delete', methods=['POST'])
@login_required
def delete_course(course_id):
    """Delete a course and all its papers and documents"""
    course = Course.query.get_or_404(course_id)
    
    # Check if the course belongs to the current user
    if course.user_id != current_user.id:
        flash('You do not have permission to delete this course.', 'danger')
        return redirect(url_for('programs.list_programs'))
    
    try:
        # Get the program_id before deleting the course
        program_id = course.program_id
        
        # Due to cascade delete, all associated papers and documents will be deleted
        db.session.delete(course)
        db.session.commit()
        flash('Course and all associated items have been deleted successfully.', 'success')
        
        return redirect(url_for('programs.view_program', program_id=program_id))
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while deleting the course: {str(e)}', 'error')
        return redirect(url_for('programs.list_programs'))

# Paper routes within courses
@programs.route('/courses/<int:course_id>/papers/new', methods=['GET', 'POST'])
@login_required
def create_paper(course_id):
    """Create a new paper within a course"""
    course = Course.query.get_or_404(course_id)
    
    # Check if the course belongs to the current user
    if course.user_id != current_user.id:
        flash('You do not have permission to add papers to this course.', 'danger')
        return redirect(url_for('programs.list_programs'))
    
    form = PaperForm()
    form.course_id.choices = [(course.id, course.name)]
    form.course_id.data = course.id
    
    if form.validate_on_submit():
        paper = Paper(
            name=form.name.data,
            paper_code=form.paper_code.data,
            description=form.description.data,
            course_id=form.course_id.data,
            user_id=current_user.id
        )
        
        db.session.add(paper)
        db.session.commit()
        
        flash('Paper created successfully!', 'success')
        return redirect(url_for('programs.view_course', course_id=course.id))
    
    return render_template('programs/paper_form.html', form=form, course=course, title='New Paper')

@programs.route('/papers/<int:paper_id>')
@login_required
def view_paper(paper_id):
    """View paper details and associated documents"""
    paper = Paper.query.get_or_404(paper_id)
    
    # Check if the paper belongs to the current user
    if paper.user_id != current_user.id:
        flash('You do not have permission to view this paper.', 'danger')
        return redirect(url_for('programs.list_programs'))
    
    # Get all documents for this paper
    from app.models.document import Document
    documents = Document.query.filter_by(paper_id=paper_id).all()
    
    return render_template('programs/view_paper.html', paper=paper, documents=documents)

@programs.route('/papers/<int:paper_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_paper(paper_id):
    """Edit paper details"""
    paper = Paper.query.get_or_404(paper_id)
    
    # Check if the paper belongs to the current user
    if paper.user_id != current_user.id:
        flash('You do not have permission to edit this paper.', 'danger')
        return redirect(url_for('programs.list_programs'))
    
    form = PaperForm()
    
    # Get all courses for the current user
    user_courses = Course.query.filter_by(user_id=current_user.id).all()
    form.course_id.choices = [(c.id, c.name) for c in user_courses]
    
    if form.validate_on_submit():
        paper.name = form.name.data
        paper.paper_code = form.paper_code.data
        paper.description = form.description.data
        paper.course_id = form.course_id.data
        
        db.session.commit()
        
        flash('Paper updated successfully!', 'success')
        return redirect(url_for('programs.view_paper', paper_id=paper.id))
    
    elif request.method == 'GET':
        form.name.data = paper.name
        form.paper_code.data = paper.paper_code
        form.description.data = paper.description
        form.course_id.data = paper.course_id
    
    return render_template('programs/paper_form.html', form=form, title='Edit Paper')

@programs.route('/papers/<int:paper_id>/delete', methods=['POST'])
@login_required
def delete_paper(paper_id):
    """Delete a paper and all its documents"""
    paper = Paper.query.get_or_404(paper_id)
    
    # Check if the paper belongs to the current user
    if paper.user_id != current_user.id:
        flash('You do not have permission to delete this paper.', 'danger')
        return redirect(url_for('programs.list_programs'))
    
    try:
        # Get the course_id before deleting the paper
        course_id = paper.course_id
        
        # Due to cascade delete, all associated documents will be deleted
        db.session.delete(paper)
        db.session.commit()
        flash('Paper and all associated documents have been deleted successfully.', 'success')
        
        return redirect(url_for('programs.view_course', course_id=course_id))
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while deleting the paper: {str(e)}', 'error')
        return redirect(url_for('programs.list_programs'))

# AJAX routes for dynamic form population
@programs.route('/api/get_courses/<int:program_id>')
@login_required
def get_courses(program_id):
    """Get courses for a specific program (for AJAX calls)"""
    courses = Course.query.filter_by(program_id=program_id, user_id=current_user.id).all()
    course_list = [{'id': c.id, 'name': c.name} for c in courses]
    return jsonify(course_list)

@programs.route('/api/get_papers/<int:course_id>')
@login_required
def get_papers(course_id):
    """Get papers for a specific course (for AJAX calls)"""
    papers = Paper.query.filter_by(course_id=course_id, user_id=current_user.id).all()
    paper_list = [{'id': p.id, 'name': p.name, 'paper_code': p.paper_code} for p in papers]
    return jsonify(paper_list)

@programs.route('/api/get_papers_by_program/<int:program_id>')
@login_required
def get_papers_by_program(program_id):
    """Get all papers for courses in a specific program (for AJAX calls)"""
    # First get all courses in this program
    courses = Course.query.filter_by(program_id=program_id, user_id=current_user.id).all()
    course_ids = [course.id for course in courses]
    
    # Then get all papers for these courses
    papers = Paper.query.filter(Paper.course_id.in_(course_ids), Paper.user_id == current_user.id).all()
    paper_list = [{'id': p.id, 'name': p.name, 'paper_code': p.paper_code, 'course_id': p.course_id} for p in papers]
    return jsonify(paper_list)
