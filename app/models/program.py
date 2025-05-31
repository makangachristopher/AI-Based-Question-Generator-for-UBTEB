from datetime import datetime
from app import db

class Program(db.Model):
    """Program model for organizing courses"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationships
    courses = db.relationship('Course', backref='program', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"Program('{self.name}')"
