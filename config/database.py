"""
Database configuration and models for Aviator Predictor
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class GameResult(db.Model):
    """Game result data model"""
    __tablename__ = 'game_results'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    multiplier = db.Column(db.Float, nullable=False)
    platform = db.Column(db.String(50), nullable=False)
    source = db.Column(db.String(100), nullable=False)
    round_id = db.Column(db.String(100))
    players_count = db.Column(db.Integer)
    total_bet = db.Column(db.Float)
    metadata = db.Column(db.Text)  # JSON string for additional data
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'multiplier': self.multiplier,
            'platform': self.platform,
            'source': self.source,
            'round_id': self.round_id,
            'players_count': self.players_count,
            'total_bet': self.total_bet,
            'metadata': json.loads(self.metadata) if self.metadata else None
        }

class Prediction(db.Model):
    """Prediction data model"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    predicted_multiplier = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    model_used = db.Column(db.String(100), nullable=False)
    actual_result = db.Column(db.Float)
    accuracy = db.Column(db.Float)
    features_used = db.Column(db.Text)  # JSON string
    prediction_context = db.Column(db.Text)  # JSON string
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'predicted_multiplier': self.predicted_multiplier,
            'confidence': self.confidence,
            'model_used': self.model_used,
            'actual_result': self.actual_result,
            'accuracy': self.accuracy,
            'features_used': json.loads(self.features_used) if self.features_used else None,
            'prediction_context': json.loads(self.prediction_context) if self.prediction_context else None
        }

class APIEndpoint(db.Model):
    """API endpoint tracking model"""
    __tablename__ = 'api_endpoints'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    url = db.Column(db.String(500), nullable=False)
    platform = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), default='active')
    last_checked = db.Column(db.DateTime, default=datetime.utcnow)
    response_time = db.Column(db.Float)
    success_rate = db.Column(db.Float)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'url': self.url,
            'platform': self.platform,
            'status': self.status,
            'last_checked': self.last_checked.isoformat() if self.last_checked else None,
            'response_time': self.response_time,
            'success_rate': self.success_rate
        }

class AnalysisPattern(db.Model):
    """Analysis pattern storage model"""
    __tablename__ = 'analysis_patterns'
    
    id = db.Column(db.Integer, primary_key=True)
    pattern_type = db.Column(db.String(50), nullable=False)
    pattern_data = db.Column(db.Text, nullable=False)  # JSON string
    confidence_score = db.Column(db.Float)
    discovered_at = db.Column(db.DateTime, default=datetime.utcnow)
    frequency = db.Column(db.Integer, default=1)
    last_occurrence = db.Column(db.DateTime)
    
    def to_dict(self):
        return {
            'id': self.id,
            'pattern_type': self.pattern_type,
            'pattern_data': json.loads(self.pattern_data) if self.pattern_data else None,
            'confidence_score': self.confidence_score,
            'discovered_at': self.discovered_at.isoformat(),
            'frequency': self.frequency,
            'last_occurrence': self.last_occurrence.isoformat() if self.last_occurrence else None
        }

class SystemLog(db.Model):
    """System logging model"""
    __tablename__ = 'system_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    level = db.Column(db.String(20), nullable=False)  # INFO, WARNING, ERROR
    module = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    details = db.Column(db.Text)  # JSON string for additional details
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'level': self.level,
            'module': self.module,
            'message': self.message,
            'details': json.loads(self.details) if self.details else None
        }

def init_database(app):
    """Initialize database with the Flask app"""
    db.init_app(app)
    
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Add some initial data if tables are empty
        if GameResult.query.count() == 0:
            _add_sample_data()
            
def _add_sample_data():
    """Add sample data for demonstration"""
    import random
    from datetime import timedelta
    
    # Add sample game results
    for i in range(100):
        timestamp = datetime.utcnow() - timedelta(minutes=i)
        
        # Generate realistic multipliers
        if random.random() < 0.7:
            multiplier = round(random.uniform(1.0, 3.0), 2)
        elif random.random() < 0.9:
            multiplier = round(random.uniform(3.0, 10.0), 2)
        else:
            multiplier = round(random.uniform(10.0, 50.0), 2)
            
        game_result = GameResult(
            timestamp=timestamp,
            multiplier=multiplier,
            platform='demo',
            source='simulation',
            round_id=f"round_{1000 + i}",
            players_count=random.randint(50, 500),
            total_bet=round(random.uniform(1000, 50000), 2),
            metadata=json.dumps({
                'simulation': True,
                'demo_data': True
            })
        )
        db.session.add(game_result)
    
    # Add sample API endpoints
    endpoints = [
        {'name': 'Spribe API', 'url': 'https://api.spribe.co/v1/', 'platform': 'spribe'},
        {'name': 'Evolution API', 'url': 'https://api.evolution.com/v2/', 'platform': 'evolution'},
        {'name': 'Pragmatic API', 'url': 'https://api.pragmaticplay.com/v1/', 'platform': 'pragmatic'}
    ]
    
    for endpoint_data in endpoints:
        endpoint = APIEndpoint(
            name=endpoint_data['name'],
            url=endpoint_data['url'],
            platform=endpoint_data['platform'],
            status='configured',
            response_time=round(random.uniform(100, 500), 1),
            success_rate=round(random.uniform(0.85, 0.99), 3)
        )
        db.session.add(endpoint)
    
    # Commit all changes
    db.session.commit()
    print("Sample data added to database")

def get_database_stats():
    """Get database statistics"""
    try:
        stats = {
            'game_results': GameResult.query.count(),
            'predictions': Prediction.query.count(),
            'api_endpoints': APIEndpoint.query.count(),
            'analysis_patterns': AnalysisPattern.query.count(),
            'system_logs': SystemLog.query.count(),
            'latest_game_result': None,
            'latest_prediction': None
        }
        
        # Get latest entries
        latest_game = GameResult.query.order_by(GameResult.timestamp.desc()).first()
        if latest_game:
            stats['latest_game_result'] = latest_game.to_dict()
            
        latest_prediction = Prediction.query.order_by(Prediction.timestamp.desc()).first()
        if latest_prediction:
            stats['latest_prediction'] = latest_prediction.to_dict()
            
        return stats
    except Exception as e:
        return {'error': str(e)}

def cleanup_old_data(days_to_keep=7):
    """Clean up old data from database"""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        # Clean up old game results
        old_results = GameResult.query.filter(GameResult.timestamp < cutoff_date).delete()
        
        # Clean up old predictions
        old_predictions = Prediction.query.filter(Prediction.timestamp < cutoff_date).delete()
        
        # Clean up old system logs
        old_logs = SystemLog.query.filter(SystemLog.timestamp < cutoff_date).delete()
        
        db.session.commit()
        
        return {
            'success': True,
            'cleaned_game_results': old_results,
            'cleaned_predictions': old_predictions,
            'cleaned_logs': old_logs,
            'cutoff_date': cutoff_date.isoformat()
        }
    except Exception as e:
        db.session.rollback()
        return {'success': False, 'error': str(e)}