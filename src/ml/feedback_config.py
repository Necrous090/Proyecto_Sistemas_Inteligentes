# feedback_config.py - Configuración centralizada del sistema de feedback

import os
from typing import Dict, Any

# Configuración de rutas
FEEDBACK_CONFIG = {
    'directories': {
        'root': 'feedback_data',
        'pending': 'feedback_data/pending',
        'processed': 'feedback_data/processed',
        'models': 'feedback_data/models',
        'analytics': 'feedback_data/analytics'
    },
    
    'processing': {
        'batch_size': 10,
        'min_feedback_for_update': 5,
        'max_feedback_per_batch': 50,
        'retention_days': 90
    },
    
    'model': {
        'incremental_learning': True,
        'update_frequency': 'daily',  # daily, weekly, immediate
        'backup_before_update': True,
        'version_control': True
    },
    
    'compatibility': {
        'required_features': [
            'tasa_asistencia',
            'completacion_tareas', 
            'puntuacion_participacion',
            'promedio_calificaciones',
            'actividades_extracurriculares',
            'involucramiento_parental_codificado'
        ],
        'supported_risk_levels': ['Bajo', 'Medio', 'Alto', 'Faible', 'Moyen', 'Élevé']
    }
}

def validate_feedback_environment() -> Dict[str, Any]:
    """
    Valida que el entorno esté correctamente configurado para el feedback
    """
    validation_result = {
        'environment_ready': True,
        'issues': [],
        'warnings': []
    }
    
    # Verificar directorios
    for key, path in FEEDBACK_CONFIG['directories'].items():
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
                validation_result['warnings'].append(f"Directorio {path} creado")
            except Exception as e:
                validation_result['environment_ready'] = False
                validation_result['issues'].append(f"No se pudo crear {path}: {e}")
        
        # Verificar permisos de escritura
        if os.path.exists(path):
            test_file = os.path.join(path, 'test_permissions.tmp')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except Exception as e:
                validation_result['environment_ready'] = False
                validation_result['issues'].append(f"Sin permisos de escritura en {path}: {e}")
    
    return validation_result

def get_feedback_system_status() -> Dict[str, Any]:
    """
    Obtiene el estado actual del sistema de feedback
    """
    from src.ml.feedback_system import get_feedback_stats
    
    try:
        stats = get_feedback_stats()
        env_status = validate_feedback_environment()
        
        return {
            'environment': env_status,
            'statistics': stats,
            'system_healthy': env_status['environment_ready'] and len(env_status['issues']) == 0
        }
    except Exception as e:
        return {
            'environment': {'environment_ready': False, 'issues': [str(e)]},
            'statistics': {},
            'system_healthy': False
        }

def get_feedback_directory_structure() -> Dict[str, str]:
    """
    Retorna la estructura de directorios del sistema de feedback
    """
    return {
        'description': 'Estructura del sistema de feedback',
        'directories': FEEDBACK_CONFIG['directories'],
        'expected_files': {
            'pending': '*.json (feedback pendiente de procesar)',
            'processed': '*.json (feedback ya procesado)',
            'models': '*.pkl (modelos actualizados por feedback)',
            'analytics': '*.json (reportes de analytics)'
        }
    }