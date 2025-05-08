from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from app import db
from app.models.course import Course
from app.models.document import Document
from app.forms.course_forms import CourseForm

courses = Blueprint('courses', __name__)

@courses.route('/courses')
@login_required
def list_courses():
    """List user's courses"""
    user_courses = Course.query.filter_by(user_id=current_user.id).all()
    return render_template('courses/courses.html', courses=user_courses)

@courses.route('/courses/new', methods=['GET', 'POST'])
@login_required
def create_course():
    """Create a new course"""
    form = CourseForm()
    
    if form.validate_on_submit():
        course = Course(
            title=form.title.data,
            description=form.description.data,
            user_id=current_user.id
        )
        
        db.session.add(course)
        db.session.commit()
        
        flash('Course created successfully!', 'success')
        return redirect(url_for('courses.list_courses'))
    
    return render_template('courses/course_form.html', form=form, title='New Course')

@courses.route('/courses/<int:course_id>')
@login_required
def view_course(course_id):
    """View course details and associated documents"""
    course = Course.query.get_or_404(course_id)
    
    # Check if the course belongs to the current user
    if course.user_id != current_user.id:
        flash('You do not have permission to view this course.', 'danger')
        return redirect(url_for('courses.list_courses'))
    
    # Get all documents for this course
    documents = Document.query.filter_by(course_id=course_id).all()
    
    return render_template('courses/view_course.html', course=course, documents=documents)

@courses.route('/courses/<int:course_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_course(course_id):
    """Edit course details"""
    course = Course.query.get_or_404(course_id)
    
    # Check if the course belongs to the current user
    if course.user_id != current_user.id:
        flash('You do not have permission to edit this course.', 'danger')
        return redirect(url_for('courses.list_courses'))
    
    form = CourseForm()
    
    if form.validate_on_submit():
        course.title = form.title.data
        course.description = form.description.data
        
        db.session.commit()
        
        flash('Course updated successfully!', 'success')
        return redirect(url_for('courses.view_course', course_id=course.id))
    
    elif request.method == 'GET':
        form.title.data = course.title
        form.description.data = course.description
    
    return render_template('courses/course_form.html', form=form, title='Edit Course')

@courses.route('/courses/<int:course_id>/delete', methods=['POST'])
@login_required
def delete_course(course_id):
    """Delete a course and optionally its documents"""
    course = Course.query.get_or_404(course_id)
    
    # Check if the course belongs to the current user
    if course.user_id != current_user.id:
        flash('You do not have permission to delete this course.', 'danger')
        return redirect(url_for('courses.list_courses'))
    
    try:
        db.session.delete(course)
        db.session.commit()
        flash('Course and associated documents have been deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while deleting the course: {str(e)}', 'error')
    
    return redirect(url_for('courses.list_courses')) 