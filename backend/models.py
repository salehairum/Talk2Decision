from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Decision(db.Model):
    __tablename__ = 'decisions'
    
    id = db.Column(db.Integer, primary_key=True)
    query = db.Column(db.String(500), nullable=False)
    extracted_decision = db.Column(db.Text, nullable=False)
    confidence = db.Column(db.String(20))  # High, Medium, Low
    file_id = db.Column(db.String(100), nullable=False)  # First source file
    source_files = db.Column(db.Text, default='')  # JSON list of all file_ids this was updated from
    status = db.Column(db.String(50), default='Open')  # Open, In-Progress, Closed
    owner = db.Column(db.String(100))  # Name/email of person responsible
    priority = db.Column(db.String(20), default='Medium')  # High, Medium, Low
    category = db.Column(db.String(100), default='General')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    evidence = db.relationship('DecisionEvidence', backref='decision', cascade='all, delete-orphan')
    action_items = db.relationship('ActionItem', backref='decision', cascade='all, delete-orphan')
    history = db.relationship('DecisionHistory', backref='decision', cascade='all, delete-orphan')
    stakeholders = db.relationship('Stakeholder', backref='decision', cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'query': self.query,
            'decision': self.extracted_decision,
            'confidence': self.confidence,
            'file_id': self.file_id,
            'source_files': self.source_files,
            'status': self.status,
            'owner': self.owner,
            'priority': self.priority,
            'category': self.category,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'evidence': [e.to_dict() for e in self.evidence],
            'action_items': [a.to_dict() for a in self.action_items],
            'stakeholders': [s.to_dict() for s in self.stakeholders]
        }


class DecisionEvidence(db.Model):
    __tablename__ = 'decision_evidence'
    
    id = db.Column(db.Integer, primary_key=True)
    decision_id = db.Column(db.Integer, db.ForeignKey('decisions.id'), nullable=False)
    user = db.Column(db.String(100), nullable=False)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.String(100))
    source_file = db.Column(db.String(100))
    
    def to_dict(self):
        return {
            'user': self.user,
            'text': self.text,
            'timestamp': self.timestamp,
            'source_file': self.source_file
        }


class ActionItem(db.Model):
    __tablename__ = 'action_items'
    
    id = db.Column(db.Integer, primary_key=True)
    decision_id = db.Column(db.Integer, db.ForeignKey('decisions.id'), nullable=False)
    task = db.Column(db.Text, nullable=False)
    owner = db.Column(db.String(100))
    due_date = db.Column(db.String(100))
    status = db.Column(db.String(50), default='Open')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'task': self.task,
            'owner': self.owner,
            'due_date': self.due_date,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class DecisionHistory(db.Model):
    __tablename__ = 'decision_history'
    
    id = db.Column(db.Integer, primary_key=True)
    decision_id = db.Column(db.Integer, db.ForeignKey('decisions.id'), nullable=False)
    field_name = db.Column(db.String(100))
    old_value = db.Column(db.Text)
    new_value = db.Column(db.Text)
    changed_at = db.Column(db.DateTime, default=datetime.utcnow)
    changed_by = db.Column(db.String(100))
    
    def to_dict(self):
        return {
            'field': self.field_name,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'changed_at': self.changed_at.isoformat() if self.changed_at else None,
            'changed_by': self.changed_by
        }


class Stakeholder(db.Model):
    __tablename__ = 'stakeholders'
    
    id = db.Column(db.Integer, primary_key=True)
    decision_id = db.Column(db.Integer, db.ForeignKey('decisions.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100))
    role = db.Column(db.String(100))
    
    def to_dict(self):
        return {
            'name': self.name,
            'email': self.email,
            'role': self.role
        }
