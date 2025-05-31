from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField, SelectField, FieldList, FormField
from wtforms.validators import DataRequired, Length, Optional

class ProgramForm(FlaskForm):
    """Program creation and edit form"""
    name = StringField('Program Name', validators=[
        DataRequired(),
        Length(min=3, max=100)
    ])
    description = TextAreaField('Description', validators=[
        Length(max=500)
    ])
    submit = SubmitField('Save Program')

class CourseForm(FlaskForm):
    """Course creation and edit form"""
    name = StringField('Course Name', validators=[
        DataRequired(),
        Length(min=3, max=100)
    ])
    description = TextAreaField('Description', validators=[
        Length(max=500)
    ])
    program_id = SelectField('Program', coerce=int, validators=[DataRequired()])
    submit = SubmitField('Save Course')

class PaperForm(FlaskForm):
    """Paper creation and edit form"""
    name = StringField('Exam Paper Name', validators=[
        DataRequired(),
        Length(min=3, max=100)
    ])
    paper_code = StringField('Paper Code', validators=[
        DataRequired(),
        Length(min=2, max=50)
    ])
    description = TextAreaField('Description', validators=[
        Length(max=500)
    ])
    course_id = SelectField('Course', coerce=int, validators=[DataRequired()])
    submit = SubmitField('Save Paper')
