import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

class ContinuousLearningSystem:
    """Sistema de aprendizaje continuo a partir del feedback de usuarios - VERSIÓN CORREGIDA"""
    
    def __init__(self, feedback_dir: str = 'feedback_data'):
        self.feedback_dir = feedback_dir
        self.ensure_directories()
        self.performance_metrics = {
            'total_feedback': 0,
            'with_corrections': 0,
            'average_rating': 0.0,
            'model_improvements': []
        }
    
    def ensure_directories(self):
        """Asegura que existan los directorios para feedback con manejo robusto de errores"""
        try:
            directories = [
                self.feedback_dir,
                f"{self.feedback_dir}/pending",
                f"{self.feedback_dir}/processed", 
                f"{self.feedback_dir}/models",
                f"{self.feedback_dir}/analytics"
            ]
            
            for directory in directories:
                try:
                    os.makedirs(directory, exist_ok=True)
                    # Verificar permisos de escritura
                    test_file = os.path.join(directory, 'test_write.tmp')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                except Exception as e:
                    logger.error(f"❌ Error con directorio {directory}: {e}")
                    raise
            
            logger.info("📁 Directorios de feedback verificados y con permisos de escritura")
            
        except Exception as e:
            logger.error(f"❌ Error crítico en ensure_directories: {e}")
            raise
    
    def save_feedback(self, student_data: Dict, prediction: Dict, 
                     user_correction: Optional[str] = None, 
                     user_notes: str = "", user_rating: Optional[int] = None) -> str:
        """
        Guarda el feedback del usuario para entrenamiento futuro - VERSIÓN CORREGIDA
        """
        try:
            # Validar datos de entrada críticos
            if not student_data or not prediction:
                logger.error("❌ Datos de estudiante o predicción inválidos")
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            feedback_id = f"feedback_{timestamp}"
            
            feedback_record = {
                'feedback_id': feedback_id,
                'timestamp': datetime.now().isoformat(),
                'student_data': student_data,
                'original_prediction': {
                    'predicted_risk': prediction.get('predicted_risk', 'Unknown'),
                    'confidence': prediction.get('confidence', 0),
                    'risk_probabilities': prediction.get('risk_probabilities', {})
                },
                'user_correction': user_correction,
                'user_rating': user_rating,
                'user_notes': user_notes,
                'used_for_training': False,
                'processed': False
            }
            
            # Validación de directorio con manejo de errores
            pending_dir = os.path.join(self.feedback_dir, 'pending')
            try:
                os.makedirs(pending_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"❌ Error creando directorio {pending_dir}: {e}")
                return None
            
            # Guardar feedback pendiente
            filename = os.path.join(pending_dir, f"{feedback_id}.json")
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(feedback_record, f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Feedback guardado exitosamente: {filename}")
            except Exception as e:
                logger.error(f"❌ Error escribiendo archivo de feedback: {e}")
                return None
            
            # Actualizar métricas
            self._update_feedback_metrics(user_correction, user_rating)
            
            return feedback_id
            
        except Exception as e:
            logger.error(f"❌ Error crítico en save_feedback: {e}")
            return None
    
    def _update_feedback_metrics(self, user_correction: Optional[str], user_rating: Optional[int]):
        """Actualiza métricas de feedback de manera segura"""
        try:
            self.performance_metrics['total_feedback'] += 1
            
            if user_correction:
                self.performance_metrics['with_corrections'] += 1
                
            if user_rating is not None:
                total_ratings = self.performance_metrics.get('total_ratings', 0)
                current_avg = self.performance_metrics.get('average_rating', 0.0)
                
                if total_ratings == 0:
                    self.performance_metrics['average_rating'] = user_rating
                else:
                    self.performance_metrics['average_rating'] = (
                        (current_avg * total_ratings + user_rating) / (total_ratings + 1)
                    )
                self.performance_metrics['total_ratings'] = total_ratings + 1
        except Exception as e:
            logger.error(f"⚠️ Error actualizando métricas: {e}")
    
    def get_pending_feedback(self) -> List[Dict]:
        """Obtiene todos los feedbacks pendientes de procesar"""
        pending_feedback = []
        pending_dir = f"{self.feedback_dir}/pending"
        
        if os.path.exists(pending_dir):
            for file in os.listdir(pending_dir):
                if file.endswith('.json'):
                    try:
                        with open(os.path.join(pending_dir, file), 'r', encoding='utf-8') as f:
                            feedback = json.load(f)
                        pending_feedback.append(feedback)
                    except Exception as e:
                        logger.error(f"❌ Error leyendo feedback {file}: {e}")
                        continue
        
        return pending_feedback
    
    def prepare_training_data(self, feedback_list: List[Dict], le_risk) -> tuple:
        """Prepara datos de entrenamiento a partir del feedback"""
        training_data = []
        valid_feedback = []
        
        for feedback in feedback_list:
            try:
                student_data = feedback['student_data']
                user_correction = feedback.get('user_correction')
                
                # Solo procesar si hay corrección del usuario y es válida
                if user_correction and user_correction in le_risk.classes_:
                    # Mapear engagement parental - compatibilidad con español
                    engagement_mapping = {
                        'Bajo': 0, 'Medio': 1, 'Alto': 2,
                        'Faible': 0, 'Moyenne': 1, 'Élevée': 2
                    }
                    
                    # Obtener valor de engagement con fallback
                    parental_engagement = student_data.get('involucramiento_parental')
                    if parental_engagement not in engagement_mapping:
                        logger.warning(f"Valor de engagement parental no reconocido: {parental_engagement}")
                        parental_engagement = 'Medio'  # Fallback
                    
                    training_example = {
                        'tasa_asistencia': float(student_data['tasa_asistencia']),
                        'completacion_tareas': float(student_data['completacion_tareas']),
                        'puntuacion_participacion': float(student_data['puntuacion_participacion']),
                        'promedio_calificaciones': float(student_data['promedio_calificaciones']),
                        'actividades_extracurriculares': int(student_data['actividades_extracurriculares']),
                        'involucramiento_parental_codificado': engagement_mapping[parental_engagement],
                        'nivel_riesgo_codificado': le_risk.transform([user_correction])[0]
                    }
                    training_data.append(training_example)
                    valid_feedback.append(feedback)
                    
            except Exception as e:
                logger.error(f"❌ Error preparando datos de feedback: {e}")
                continue
        
        return training_data, valid_feedback
    
    def process_feedback_batch(self, model, le_risk, scaler, batch_size: int = 10) -> Dict[str, Any]:
        """
        Procesa un lote de feedbacks para actualizar el modelo - VERSIÓN CORREGIDA
        """
        try:
            # Validar entradas críticas
            if model is None or le_risk is None:
                return {'processed': 0, 'error': 'Modelo o LabelEncoder no disponibles', 'model_updated': False}
            
            pending_feedback = self.get_pending_feedback()
            if not pending_feedback:
                return {'processed': 0, 'message': 'No hay feedback pendiente', 'model_updated': False}
            
            logger.info(f"🔄 Procesando {len(pending_feedback)} feedbacks pendientes...")
            
            # Preparar datos de entrenamiento con validación
            training_data, valid_feedback = self.prepare_training_data(pending_feedback, le_risk)
            
            if not training_data:
                logger.info("No hay feedback válido para entrenamiento (sin correcciones de usuario)")
                # Mover feedback sin correcciones a procesados
                self._move_feedback_without_corrections(pending_feedback)
                return {'processed': 0, 'message': 'No hay feedback con correcciones válidas', 'model_updated': False}
            
            # Limitar al tamaño del lote
            training_data = training_data[:batch_size]
            valid_feedback = valid_feedback[:batch_size]
            
            # Convertir a DataFrame con manejo de errores
            try:
                df_new = pd.DataFrame(training_data)
                features = ['tasa_asistencia', 'completacion_tareas', 'puntuacion_participacion', 
                           'promedio_calificaciones', 'actividades_extracurriculares', 'involucramiento_parental_codificado']
                
                # Verificar que tenemos todas las características necesarias
                missing_features = [f for f in features if f not in df_new.columns]
                if missing_features:
                    logger.error(f"Faltan características: {missing_features}")
                    return {'processed': 0, 'error': f'Faltan características: {missing_features}', 'model_updated': False}
                
                X_new = df_new[features]
                y_new = df_new['nivel_riesgo_codificado']
                
            except Exception as e:
                logger.error(f"Error preparando datos de entrenamiento: {e}")
                return {'processed': 0, 'error': f'Error en datos de entrenamiento: {e}', 'model_updated': False}
            
            # Realizar aprendizaje incremental
            updated_model = self.incremental_learning(model, X_new, y_new)
            
            # Mover feedback procesado
            self._move_processed_feedback(valid_feedback)
            
            # Guardar el modelo actualizado si es diferente
            if updated_model != model:
                model_version = f"model_continuous_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                model_path = os.path.join(self.feedback_dir, 'models', model_version)
                
                try:
                    model_data = {
                        'model': updated_model,
                        'feedback_count': len(training_data),
                        'training_date': datetime.now().isoformat(),
                        'features': features,
                        'label_encoder': le_risk
                    }
                    
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    joblib.dump(model_data, model_path)
                    
                    # Actualizar el modelo principal
                    self._update_main_model(updated_model, le_risk)
                    
                    # Registrar mejora del modelo
                    accuracy_change = self._evaluate_model_change(model, updated_model, X_new, y_new)
                    self.performance_metrics['model_improvements'].append({
                        'timestamp': datetime.now().isoformat(),
                        'samples_added': len(training_data),
                        'accuracy_change': accuracy_change,
                        'model_version': model_version
                    })
                    
                    logger.info(f"✅ Modelo actualizado con {len(training_data)} nuevos ejemplos")
                    
                    return {
                        'processed': len(training_data),
                        'model_updated': True,
                        'new_samples': len(training_data),
                        'model_path': model_path,
                        'accuracy_change': accuracy_change
                    }
                    
                except Exception as e:
                    logger.error(f"Error guardando modelo actualizado: {e}")
                    return {'processed': 0, 'error': f'Error guardando modelo: {e}', 'model_updated': False}
            else:
                logger.info("Modelo no cambió después del aprendizaje incremental")
                return {'processed': len(training_data), 'model_updated': False, 'message': 'Modelo sin cambios'}
                
        except Exception as e:
            logger.error(f"❌ Error en process_feedback_batch: {e}")
            return {'processed': 0, 'error': str(e), 'model_updated': False}
    
    def _move_feedback_without_corrections(self, feedback_list: List[Dict]):
        """Mueve feedback sin correcciones a procesados"""
        for feedback in feedback_list:
            try:
                if not feedback.get('user_correction'):
                    filename = f"{feedback['feedback_id']}.json"
                    old_path = f"{self.feedback_dir}/pending/{filename}"
                    new_path = f"{self.feedback_dir}/processed/{filename}"
                    
                    if os.path.exists(old_path):
                        feedback['processed'] = True
                        feedback['processed_date'] = datetime.now().isoformat()
                        feedback['used_for_training'] = False
                        feedback['processing_notes'] = 'Sin corrección de usuario'
                        
                        with open(new_path, 'w', encoding='utf-8') as f:
                            json.dump(feedback, f, indent=2, ensure_ascii=False)
                        os.remove(old_path)
                        
            except Exception as e:
                logger.error(f"❌ Error moviendo feedback sin correcciones: {e}")
    
    def _move_processed_feedback(self, feedback_list: List[Dict]):
        """Mueve feedback procesado a la carpeta correspondiente"""
        for feedback in feedback_list:
            try:
                filename = f"{feedback['feedback_id']}.json"
                old_path = f"{self.feedback_dir}/pending/{filename}"
                new_path = f"{self.feedback_dir}/processed/{filename}"
                
                if os.path.exists(old_path):
                    feedback['processed'] = True
                    feedback['processed_date'] = datetime.now().isoformat()
                    feedback['used_for_training'] = True
                    
                    with open(new_path, 'w', encoding='utf-8') as f:
                        json.dump(feedback, f, indent=2, ensure_ascii=False)
                    os.remove(old_path)
                    
            except Exception as e:
                logger.error(f"❌ Error moviendo feedback {feedback.get('feedback_id')}: {e}")
    
    def _update_main_model(self, updated_model, le_risk):
        """Actualiza el modelo principal del sistema"""
        try:
            # Guardar como el modelo principal más reciente
            main_model_path = "models/student_risk_model_latest.pkl"
            os.makedirs("models", exist_ok=True)
            
            model_data = {
                'model': updated_model,
                'label_encoder': le_risk,
                'last_updated': datetime.now().isoformat(),
                'version': 'continuous_learning',
                'update_type': 'feedback_incremental'
            }
            
            joblib.dump(model_data, main_model_path)
            logger.info(f"💾 Modelo principal actualizado: {main_model_path}")
            
        except Exception as e:
            logger.error(f"❌ Error actualizando modelo principal: {e}")
    
    def _evaluate_model_change(self, old_model, new_model, X_test, y_test) -> float:
        """Evalúa el cambio en el modelo"""
        try:
            old_accuracy = accuracy_score(y_test, old_model.predict(X_test))
            new_accuracy = accuracy_score(y_test, new_model.predict(X_test))
            return new_accuracy - old_accuracy
        except:
            return 0.0
    
    def incremental_learning(self, model, X_new, y_new):
        """
        Realiza aprendizaje incremental en el modelo existente
        """
        try:
            logger.info("🔄 Realizando aprendizaje incremental...")
            
            # Para RandomForest, creamos un nuevo modelo con parámetros similares
            if hasattr(model, 'estimators_'):
                updated_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                )
                
                # Entrenar con los nuevos datos
                updated_model.fit(X_new, y_new)
                
                return updated_model
            else:
                logger.warning("⚠️ Modelo no compatible con aprendizaje incremental, retornando original")
                return model
            
        except Exception as e:
            logger.error(f"❌ Error en aprendizaje incremental: {e}")
            return model
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de feedback"""
        pending_dir = f"{self.feedback_dir}/pending"
        processed_dir = f"{self.feedback_dir}/processed"
        models_dir = f"{self.feedback_dir}/models"
        
        stats = {
            'pending_feedback': 0,
            'processed_feedback': 0,
            'total_feedback': 0,
            'with_corrections': 0,
            'model_versions': 0,
            'last_processed': None,
            'performance_metrics': self.performance_metrics
        }
        
        # Contar feedback pendiente
        if os.path.exists(pending_dir):
            stats['pending_feedback'] = len([f for f in os.listdir(pending_dir) if f.endswith('.json')])
        
        # Contar feedback procesado y con correcciones
        if os.path.exists(processed_dir):
            processed_files = [f for f in os.listdir(processed_dir) if f.endswith('.json')]
            stats['processed_feedback'] = len(processed_files)
            
            # Encontrar fecha del último procesamiento y contar correcciones
            latest_date = None
            for file in processed_files:
                try:
                    with open(os.path.join(processed_dir, file), 'r', encoding='utf-8') as f:
                        feedback = json.load(f)
                    if feedback.get('user_correction'):
                        stats['with_corrections'] += 1
                    
                    # Encontrar la fecha más reciente
                    processed_date = feedback.get('processed_date')
                    if processed_date:
                        date_obj = datetime.fromisoformat(processed_date.replace('Z', '+00:00'))
                        if latest_date is None or date_obj > latest_date:
                            latest_date = date_obj
                except:
                    continue
            
            stats['last_processed'] = latest_date.isoformat() if latest_date else None
        
        # Contar versiones de modelo
        if os.path.exists(models_dir):
            stats['model_versions'] = len([f for f in os.listdir(models_dir) if f.endswith('.pkl')])
        
        stats['total_feedback'] = stats['pending_feedback'] + stats['processed_feedback']
        
        return stats
    
    def get_recent_feedback(self, limit: int = 5) -> List[Dict]:
        """Obtiene feedback reciente para mostrar en la interfaz"""
        all_feedback = []
        pending_dir = f"{self.feedback_dir}/pending"
        processed_dir = f"{self.feedback_dir}/processed"
        
        # Obtener de pendientes
        if os.path.exists(pending_dir):
            for file in sorted(os.listdir(pending_dir), reverse=True):
                if file.endswith('.json'):
                    try:
                        with open(os.path.join(pending_dir, file), 'r', encoding='utf-8') as f:
                            feedback = json.load(f)
                        feedback['status'] = 'pending'
                        all_feedback.append(feedback)
                    except:
                        continue
        
        # Obtener de procesados
        if os.path.exists(processed_dir):
            for file in sorted(os.listdir(processed_dir), reverse=True):
                if file.endswith('.json'):
                    try:
                        with open(os.path.join(processed_dir, file), 'r', encoding='utf-8') as f:
                            feedback = json.load(f)
                        feedback['status'] = 'processed'
                        all_feedback.append(feedback)
                    except:
                        continue
        
        # Ordenar por timestamp y limitar
        all_feedback.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return all_feedback[:limit]
    
    def create_feedback_analytics(self) -> Dict[str, Any]:
        """Crea analytics detallados del sistema de feedback"""
        analytics = {
            'summary': self.get_feedback_stats(),
            'timeline_data': self._get_timeline_data(),
            'rating_distribution': self._get_rating_distribution(),
            'improvement_trends': self._get_improvement_trends()
        }
        
        # Guardar analytics
        try:
            analytics_file = f"{self.feedback_dir}/analytics/feedback_analytics_{datetime.now().strftime('%Y%m%d')}.json"
            with open(analytics_file, 'w', encoding='utf-8') as f:
                json.dump(analytics, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando analytics: {e}")
        
        return analytics
    
    def _get_timeline_data(self) -> List[Dict]:
        """Obtiene datos para timeline de feedback"""
        try:
            # Implementación real basada en archivos existentes
            feedback_files = []
            for status in ['pending', 'processed']:
                dir_path = f"{self.feedback_dir}/{status}"
                if os.path.exists(dir_path):
                    for file in os.listdir(dir_path):
                        if file.endswith('.json'):
                            feedback_files.append((dir_path, file))
            
            # Agrupar por fecha
            date_counts = {}
            for dir_path, file in feedback_files:
                try:
                    with open(os.path.join(dir_path, file), 'r', encoding='utf-8') as f:
                        feedback = json.load(f)
                    date_str = feedback.get('timestamp', '').split('T')[0]
                    if date_str:
                        date_counts[date_str] = date_counts.get(date_str, 0) + 1
                except:
                    continue
            
            timeline_data = [{'date': date, 'feedback_count': count} 
                           for date, count in sorted(date_counts.items())]
            
            return timeline_data[-7:]  # Últimos 7 días
            
        except Exception as e:
            logger.error(f"Error obteniendo timeline data: {e}")
            return []
    
    def _get_rating_distribution(self) -> Dict[str, int]:
        """Obtiene distribución de ratings"""
        try:
            rating_counts = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
            
            # Contar ratings de archivos procesados
            processed_dir = f"{self.feedback_dir}/processed"
            if os.path.exists(processed_dir):
                for file in os.listdir(processed_dir):
                    if file.endswith('.json'):
                        try:
                            with open(os.path.join(processed_dir, file), 'r', encoding='utf-8') as f:
                                feedback = json.load(f)
                            rating = feedback.get('user_rating')
                            if rating and 1 <= rating <= 5:
                                rating_counts[str(rating)] += 1
                        except:
                            continue
            
            return rating_counts
            
        except Exception as e:
            logger.error(f"Error obteniendo distribución de ratings: {e}")
            return {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    
    def _get_improvement_trends(self) -> List[Dict]:
        """Obtiene tendencias de mejora del modelo"""
        return self.performance_metrics.get('model_improvements', [])

# Instancia global del sistema
feedback_system = ContinuousLearningSystem()

def save_user_feedback(student_data: Dict, prediction: Dict, 
                      user_correction: Optional[str] = None, 
                      user_notes: str = "", user_rating: Optional[int] = None) -> str:
    """
    Función helper para guardar feedback del usuario
    """
    return feedback_system.save_feedback(student_data, prediction, user_correction, user_notes, user_rating)

def process_feedback(model, le_risk, scaler, batch_size: int = 10) -> Dict[str, Any]:
    """
    Función helper para procesar feedback pendiente
    """
    return feedback_system.process_feedback_batch(model, le_risk, scaler, batch_size)

def get_feedback_stats() -> Dict[str, Any]:
    """Función helper para obtener estadísticas"""
    return feedback_system.get_feedback_stats()

def get_recent_feedback(limit: int = 5) -> List[Dict]:
    """Función helper para obtener feedback reciente"""
    return feedback_system.get_recent_feedback(limit)

def get_feedback_analytics() -> Dict[str, Any]:
    """Función helper para obtener analytics de feedback"""
    return feedback_system.create_feedback_analytics()

# 🔧 NUEVA: Función de diagnóstico del sistema
def debug_feedback_system():
    """Provee diagnóstico completo del sistema de feedback"""
    import streamlit as st
    
    diagnosis = {
        'directories': {},
        'file_counts': {},
        'system_status': {},
        'test_results': {}
    }
    
    # Verificar directorios
    directories = ['feedback_data', 'feedback_data/pending', 'feedback_data/processed', 
                  'feedback_data/models', 'feedback_data/analytics', 'models']
    
    for dir_path in directories:
        exists = os.path.exists(dir_path)
        writable = os.access(dir_path, os.W_OK) if exists else False
        diagnosis['directories'][dir_path] = {'exists': exists, 'writable': writable}
    
    # Contar archivos
    for status in ['pending', 'processed']:
        dir_path = f"feedback_data/{status}"
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
            diagnosis['file_counts'][status] = len(files)
        else:
            diagnosis['file_counts'][status] = 0
    
    # Verificar sistema
    try:
        stats = get_feedback_stats()
        diagnosis['system_status']['stats'] = stats
        diagnosis['system_status']['stats_available'] = True
    except Exception as e:
        diagnosis['system_status']['stats_error'] = str(e)
        diagnosis['system_status']['stats_available'] = False
    
    # Prueba de funcionalidad
    try:
        test_student = {
            'tasa_asistencia': 85,
            'completacion_tareas': 75,
            'puntuacion_participacion': 7.0,
            'promedio_calificaciones': 14.5,
            'actividades_extracurriculares': 2,
            'involucramiento_parental': 'Medio'
        }
        
        test_prediction = {
            'predicted_risk': 'Medio',
            'confidence': 75.5,
            'risk_probabilities': {'Bajo': 0.2, 'Medio': 0.6, 'Alto': 0.2}
        }
        
        test_id = save_user_feedback(
            test_student, 
            test_prediction, 
            user_correction='Bajo',
            user_notes="Feedback de prueba del sistema de diagnóstico",
            user_rating=5
        )
        
        diagnosis['test_results']['save_test'] = {
            'success': test_id is not None,
            'feedback_id': test_id
        }
        
    except Exception as e:
        diagnosis['test_results']['save_test'] = {
            'success': False,
            'error': str(e)
        }
    
    return diagnosis

if __name__ == "__main__":
    # Pruebas del sistema de feedback
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Probando sistema de feedback...")
    
    # Ejemplo de feedback
    sample_student = {
        'tasa_asistencia': 75,
        'completacion_tareas': 60,
        'puntuacion_participacion': 4.0,
        'promedio_calificaciones': 9.5,
        'actividades_extracurriculares': 0,
        'involucramiento_parental': 'Bajo'
    }
    
    sample_prediction = {
        'predicted_risk': 'Alto',
        'confidence': 85.5,
        'risk_probabilities': {'Bajo': 0.1, 'Medio': 0.2, 'Alto': 0.7}
    }
    
    # Guardar feedback
    feedback_id = save_user_feedback(
        sample_student, 
        sample_prediction, 
        user_correction='Medio',
        user_notes="El estudiante ha mostrado mejoría reciente",
        user_rating=4
    )
    
    print(f"✅ Feedback guardado: {feedback_id}")
    
    # Obtener estadísticas
    stats = get_feedback_stats()
    print(f"📊 Estadísticas: {stats}")
