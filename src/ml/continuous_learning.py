import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

class ContinuousLearningManager:
    """Gestor de aprendizaje continuo automático"""
    
    def __init__(self, feedback_system, model_training_module):
        self.feedback_system = feedback_system
        self.model_training = model_training_module
        self.learning_metrics = {
            'total_batches_processed': 0,
            'total_feedback_learned': 0,
            'accuracy_improvements': [],
            'last_processing_time': None,
            'model_versions_created': 0
        }
        
    def check_and_process_feedback(self, model, le_risk, scaler, batch_threshold: int = 5) -> Dict[str, Any]:
        """
        Verifica y procesa automáticamente el feedback si hay suficientes datos
        """
        try:
            # Obtener estadísticas de feedback pendiente
            stats = self.feedback_system.get_feedback_stats()
            pending_count = stats.get('pending_feedback', 0)
            
            logger.info(f"🔍 Feedback pendiente: {pending_count}, Umbral: {batch_threshold}")
            
            if pending_count >= batch_threshold:
                logger.info(f"🔄 Procesando lote automático de {pending_count} feedbacks...")
                
                # Procesar el lote
                result = self.feedback_system.process_feedback_batch(
                    model, le_risk, scaler, batch_size=pending_count
                )
                
                if result.get('model_updated', False):
                    self._update_learning_metrics(result)
                    self._create_learning_report(result)
                    
                    logger.info(f"✅ Lote procesado: {result['processed']} feedbacks")
                    return {
                        'processed': True,
                        'feedback_processed': result['processed'],
                        'model_updated': True,
                        'accuracy_change': result.get('accuracy_change', 0),
                        'new_model_version': result.get('model_path', '')
                    }
                else:
                    logger.info("ℹ️ Feedback procesado pero modelo no actualizado")
                    return {
                        'processed': True,
                        'feedback_processed': result.get('processed', 0),
                        'model_updated': False,
                        'message': result.get('message', 'No se actualizó el modelo')
                    }
            else:
                return {
                    'processed': False,
                    'pending_feedback': pending_count,
                    'needed_for_batch': batch_threshold - pending_count
                }
                
        except Exception as e:
            logger.error(f"❌ Error en procesamiento automático: {e}")
            return {'error': str(e), 'processed': False}
    
    def _update_learning_metrics(self, processing_result: Dict):
        """Actualiza las métricas de aprendizaje"""
        self.learning_metrics['total_batches_processed'] += 1
        self.learning_metrics['total_feedback_learned'] += processing_result.get('processed', 0)
        self.learning_metrics['last_processing_time'] = datetime.now().isoformat()
        self.learning_metrics['model_versions_created'] += 1
        
        # Registrar mejora de precisión si está disponible
        accuracy_change = processing_result.get('accuracy_change')
        if accuracy_change is not None:
            self.learning_metrics['accuracy_improvements'].append({
                'timestamp': datetime.now().isoformat(),
                'improvement': accuracy_change,
                'samples_added': processing_result.get('processed', 0)
            })
    
    def _create_learning_report(self, processing_result: Dict):
        """Crea un reporte de aprendizaje"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'processing_result': processing_result,
                'current_metrics': self.learning_metrics.copy(),
                'system_state': {
                    'total_feedback_learned': self.learning_metrics['total_feedback_learned'],
                    'batches_processed': self.learning_metrics['total_batches_processed'],
                    'model_versions': self.learning_metrics['model_versions_created']
                }
            }
            
            # Guardar reporte
            os.makedirs('feedback_data/analytics', exist_ok=True)
            report_file = f"feedback_data/analytics/learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            logger.info(f"📊 Reporte de aprendizaje guardado: {report_file}")
            
        except Exception as e:
            logger.error(f"Error creando reporte de aprendizaje: {e}")
    
    def get_learning_analytics(self) -> Dict[str, Any]:
        """Obtiene analytics detallados del aprendizaje continuo"""
        try:
            # 🔧 CORRECCIÓN: Usar create_feedback_analytics en lugar de get_feedback_analytics
            base_analytics = self.feedback_system.create_feedback_analytics()
        except Exception as e:
            logger.error(f"Error obteniendo analytics de feedback: {e}")
            # Fallback si hay error
            base_analytics = {
                'summary': self.feedback_system.get_feedback_stats(),
                'timeline_data': [],
                'rating_distribution': {},
                'improvement_trends': []
            }
        
        learning_analytics = {
            'continuous_learning': {
                'metrics': self.learning_metrics,
                'efficiency': self._calculate_learning_efficiency(),
                'improvement_trend': self._calculate_improvement_trend(),
                'next_processing_estimate': self._estimate_next_processing()
            },
            'feedback_analytics': base_analytics
        }
        
        return learning_analytics    
    def _calculate_learning_efficiency(self) -> Dict[str, float]:
        """Calcula la eficiencia del aprendizaje"""
        total_feedback = self.learning_metrics['total_feedback_learned']
        batches = self.learning_metrics['total_batches_processed']
        
        if batches == 0:
            return {'efficiency_score': 0.0, 'feedback_per_batch': 0.0}
        
        avg_feedback_per_batch = total_feedback / batches
        efficiency_score = min(avg_feedback_per_batch / 10.0, 1.0)  # Normalizado a 10 por lote
        
        return {
            'efficiency_score': efficiency_score,
            'feedback_per_batch': avg_feedback_per_batch,
            'utilization_rate': efficiency_score * 100
        }
    
    def _calculate_improvement_trend(self) -> Dict[str, Any]:
        """Calcula la tendencia de mejora"""
        improvements = self.learning_metrics['accuracy_improvements']
        
        if not improvements:
            return {'trend': 'stable', 'avg_improvement': 0.0, 'total_improvement': 0.0}
        
        recent_improvements = improvements[-5:]  # Últimas 5 mejoras
        avg_improvement = sum(imp['improvement'] for imp in recent_improvements) / len(recent_improvements)
        total_improvement = sum(imp['improvement'] for imp in improvements)
        
        if avg_improvement > 0.01:
            trend = 'improving'
        elif avg_improvement < -0.01:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'avg_improvement': avg_improvement,
            'total_improvement': total_improvement,
            'improvement_count': len(improvements)
        }
    
    def _estimate_next_processing(self) -> Dict[str, Any]:
        """Estima cuándo será el próximo procesamiento"""
        stats = self.feedback_system.get_feedback_stats()
        pending = stats.get('pending_feedback', 0)
        needed = max(0, 5 - pending)  # Asumiendo umbral de 5
        
        avg_feedback_per_day = self._calculate_daily_feedback_rate()
        
        if avg_feedback_per_day > 0:
            days_until_next = needed / avg_feedback_per_day
        else:
            days_until_next = float('inf')
        
        return {
            'pending_feedback': pending,
            'needed_for_next_batch': needed,
            'estimated_days_until_processing': days_until_next,
            'daily_feedback_rate': avg_feedback_per_day
        }
    
    def _calculate_daily_feedback_rate(self) -> float:
        """Calcula la tasa diaria de feedback recibido"""
        try:
            # Analizar archivos de feedback para calcular tasa
            pending_dir = 'feedback_data/pending'
            if os.path.exists(pending_dir):
                files = [f for f in os.listdir(pending_dir) if f.endswith('.json')]
                if len(files) > 1:
                    # Obtener timestamps de los archivos
                    timestamps = []
                    for file in files[:10]:  # Muestra de 10 archivos
                        try:
                            with open(os.path.join(pending_dir, file), 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                timestamp = data.get('timestamp', '')
                                if timestamp:
                                    timestamps.append(datetime.fromisoformat(timestamp.replace('Z', '+00:00')))
                        except:
                            continue
                    
                    if len(timestamps) >= 2:
                        timestamps.sort()
                        time_range = (timestamps[-1] - timestamps[0]).total_seconds() / 86400  # en días
                        if time_range > 0:
                            return len(files) / time_range
            
            return 1.0  # Valor por defecto
            
        except Exception as e:
            logger.error(f"Error calculando tasa de feedback: {e}")
            return 1.0

# Instancia global
continuous_learning_manager = None

def init_continuous_learning(feedback_system, model_training_module):
    """Inicializa el sistema de aprendizaje continuo"""
    global continuous_learning_manager
    continuous_learning_manager = ContinuousLearningManager(feedback_system, model_training_module)
    return continuous_learning_manager

def get_continuous_learning_manager():
    """Obtiene el gestor de aprendizaje continuo"""
    global continuous_learning_manager
    return continuous_learning_manager

