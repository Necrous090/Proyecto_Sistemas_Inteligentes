"""
Módulos de machine learning y recomendaciones - Versión Avanzada
"""

from .model_training import (
    train_risk_prediction_model, 
    train_advanced_risk_model,
    load_latest_model, 
    save_model,
    save_model_with_versioning,
    get_model_info
)

from .recommendation_system import (
    generate_recommendations, 
    generate_contextual_recommendations,
    generate_proactive_alerts,
    validate_student_data
)

from .feedback_system import (
    save_user_feedback, 
    process_feedback, 
    get_feedback_stats, 
    get_recent_feedback,
    get_feedback_analytics
)

__all__ = [
    # Model Training
    'train_risk_prediction_model',
    'train_advanced_risk_model', 
    'load_latest_model',
    'save_model',
    'save_model_with_versioning',
    'get_model_info',
    
    # Recommendation System
    'generate_recommendations',
    'generate_contextual_recommendations',
    'generate_proactive_alerts', 
    'validate_student_data',
    
    # Feedback System
    'save_user_feedback',
    'process_feedback',
    'get_feedback_stats',
    'get_recent_feedback',
    'get_feedback_analytics'
]

__version__ = "2.0.0"
__author__ = "Sistema de Recomendación Educativa"
__description__ = "Sistema avanzado de recomendaciones educativas con IA explicativa y aprendizaje continuo"