from datetime import datetime
from app import db

class Paper(db.Model):
    """Paper model for organizing exam papers within a course"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    paper_code = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    
    # Relationships
    documents = db.relationship('Document', backref='paper', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"Paper('{self.name}', '{self.paper_code}')"
