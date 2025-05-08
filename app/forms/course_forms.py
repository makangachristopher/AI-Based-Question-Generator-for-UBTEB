from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField
from wtforms.validators import DataRequired, Length

class CourseForm(FlaskForm):
    """Course creation and edit form"""
    title = StringField('Course Title', validators=[
        DataRequired(),
        Length(min=3, max=100)
    ])
    description = TextAreaField('Description', validators=[
        Length(max=500)
    ])
    submit = SubmitField('Save Course') 