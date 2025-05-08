from datetime import datetime
from app import db

class QuestionSet(db.Model):
    """QuestionSet model for grouping sets of questions together"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Exam metadata
    exam_series = db.Column(db.String(100), nullable=True)
    programme_list = db.Column(db.String(255), nullable=True)
    paper_name = db.Column(db.String(255), nullable=True)
    paper_code = db.Column(db.String(50), nullable=True)
    year_semester = db.Column(db.String(100), nullable=True)
    exam_date = db.Column(db.String(50), nullable=True)
    difficulty = db.Column(db.String(20), nullable=False)
    
    # Foreign keys
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationships
    questions = db.relationship('Question', backref='question_set', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"QuestionSet('{self.title}', '{self.created_at}')"

class Question(db.Model):
    """Question model for storing generated questions"""
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    options = db.Column(db.JSON, nullable=True)  # For multiple choice questions
    question_type = db.Column(db.String(20), nullable=False)  # 'section_a' or 'section_b'
    difficulty = db.Column(db.String(20), nullable=False)  # 'easy', 'medium', 'hard'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign keys
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    question_set_id = db.Column(db.Integer, db.ForeignKey('question_set.id'), nullable=False)
    
    def __repr__(self):
        return f"Question('{self.content[:30]}...', '{self.question_type}')" 