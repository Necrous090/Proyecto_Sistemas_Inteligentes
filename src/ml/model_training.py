import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
import joblib
import os
import sys
import logging
from datetime import datetime
from typing import Tuple, Optional, Any, Dict, List
import json
import matplotlib.pyplot as plt

# Configurar logging
logger = logging.getLogger(__name__)

class AdvancedModelTrainer:
    """Clase mejorada para entrenamiento y gestión de modelos de ML"""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.ensure_directories()
        self.best_model = None
        self.model_comparison = {}
        
    def ensure_directories(self):
        """Asegura que los directorios para modelos existan"""
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(f"{self.model_dir}/versions", exist_ok=True)
        os.makedirs(f"{self.model_dir}/comparison", exist_ok=True)
        logger.info(f"📁 Directorios de modelos verificados: {self.model_dir}")
    
    def get_available_models(self) -> Dict:
        """Retorna los modelos disponibles para comparación"""
        return {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15, None],
                    'min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.1, 0.05],
                    'max_depth': [3, 5]
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True, class_weight='balanced'),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
                }
            }
        }
    
    def compare_models(self, X_train, y_train, X_test, y_test, cv_folds: int = 5) -> Dict:
        """Compara múltiples modelos y selecciona el mejor"""
        logger.info("🔍 Comparando modelos...")
        
        models = self.get_available_models()
        comparison_results = {}
        
        for name, config in models.items():
            try:
                logger.info(f"🧪 Evaluando modelo: {name}")
                
                # Búsqueda de hiperparámetros
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'], 
                    cv=cv_folds, 
                    scoring='accuracy',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                
                # Evaluación
                y_pred = grid_search.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                
                # Validación cruzada
                cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=cv_folds)
                
                comparison_results[name] = {
                    'best_model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                logger.info(f"✅ {name} - Precisión: {accuracy:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error evaluando modelo {name}: {e}")
                continue
        
        # Seleccionar el mejor modelo
        if comparison_results:
            best_model_name = max(comparison_results.keys(), 
                                key=lambda x: comparison_results[x]['f1_score'])
            self.best_model = comparison_results[best_model_name]
            logger.info(f"🏆 Mejor modelo: {best_model_name} con F1: {self.best_model['f1_score']:.4f}")
        
        self.model_comparison = comparison_results
        return comparison_results
    
    def create_ensemble_model(self, models_dict: Dict) -> Any:
        """Crea un modelo ensemble con los mejores modelos"""
        from sklearn.ensemble import VotingClassifier
        
        estimators = []
        for name, results in models_dict.items():
            if 'best_model' in results:
                estimators.append((name, results['best_model']))
        
        if len(estimators) >= 2:
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            logger.info(f"🤝 Ensemble creado con {len(estimators)} modelos")
            return ensemble
        else:
            logger.warning("⚠️ No hay suficientes modelos para crear ensemble")
            return None

class ModelTrainer:
    """Clase para entrenamiento y gestión de modelos de ML"""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.ensure_directories()
        
    def ensure_directories(self):
        """Asegura que los directorios para modelos existan"""
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(f"{self.model_dir}/versions", exist_ok=True)
        logger.info(f"📁 Directorios de modelos verificados: {self.model_dir}")
    
    def prepare_data(self, X, y, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepara los datos para entrenamiento y prueba
        """
        try:
            # Validar datos de entrada
            if X is None or y is None:
                raise ValueError("X o y son None")
            
            if len(X) != len(y):
                raise ValueError("X e y tienen longitudes diferentes")
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            logger.info(f"📊 Datos divididos - Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"❌ Error preparando datos: {e}")
            raise
    
    def create_model(self, **kwargs) -> RandomForestClassifier:
        """
        Crea un modelo de Random Forest con configuración personalizable
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        
        # Actualizar parámetros con los proporcionados
        default_params.update(kwargs)
        
        model = RandomForestClassifier(**default_params)
        logger.info(f"🧠 Modelo creado con parámetros: {default_params}")
        return model
    
    def train_model(self, model, X_train, y_train) -> RandomForestClassifier:
        """
        Entrena el modelo con datos de entrenamiento
        """
        try:
            logger.info("🚀 Entrenando modelo...")
            model.fit(X_train, y_train)
            logger.info("✅ Modelo entrenado exitosamente")
            return model
        except Exception as e:
            logger.error(f"❌ Error entrenando modelo: {e}")
            raise
    
    def evaluate_model(self, model, X_test, y_test) -> Dict[str, Any]:
        """
        Evalúa el modelo y retorna métricas detalladas
        """
        try:
            logger.info("📈 Evaluando modelo...")
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            # Métricas adicionales
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            evaluation = {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'classification_report': report,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"✅ Evaluación completada - Precisión: {accuracy:.4f}")
            return evaluation
            
        except Exception as e:
            logger.error(f"❌ Error evaluando modelo: {e}")
            raise

def train_risk_prediction_model(X, y, **kwargs) -> Tuple[RandomForestClassifier, float, Dict[str, Any]]:
    """
    Entrena un modelo para predecir el nivel de riesgo académico
    """
    try:
        trainer = ModelTrainer()
        
        # Preparar datos
        X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
        
        # Crear modelo
        model = trainer.create_model(**kwargs)
        
        # Entrenar modelo
        model = trainer.train_model(model, X_train, y_train)
        
        # Evaluar modelo
        evaluation = trainer.evaluate_model(model, X_test, y_test)
        accuracy = evaluation['accuracy']
        
        # Reporte de clasificación en formato string para logging
        report_str = classification_report(y_test, evaluation['predictions'])
        
        logger.info(f"🎯 Modelo entrenado - Precisión: {accuracy:.4f}")
        logger.info(f"📊 Reporte de clasificación:\n{report_str}")
        
        return model, accuracy, evaluation
        
    except Exception as e:
        logger.error(f"❌ Error en el pipeline de entrenamiento: {e}")
        raise

def train_advanced_risk_model(X, y, test_size: float = 0.2, compare_models: bool = True) -> Tuple[Any, float, Dict]:
    """
    Entrena un modelo avanzado para predecir el nivel de riesgo académico
    """
    try:
        trainer = AdvancedModelTrainer()
        
        # Preparar datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        if compare_models:
            # Comparar múltiples modelos
            comparison = trainer.compare_models(X_train, y_train, X_test, y_test)
            
            if trainer.best_model:
                model = trainer.best_model['best_model']
                accuracy = trainer.best_model['accuracy']
                evaluation = {
                    'comparison_results': comparison,
                    'best_model_name': max(comparison.keys(), 
                                         key=lambda x: comparison[x]['f1_score']),
                    'metrics': trainer.best_model
                }
            else:
                # Fallback a Random Forest
                logger.warning("⚠️ Usando Random Forest como fallback")
                model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                model.fit(X_train, y_train)
                accuracy = accuracy_score(y_test, model.predict(X_test))
                evaluation = {'accuracy': accuracy}
        else:
            # Entrenamiento simple
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, model.predict(X_test))
            evaluation = {'accuracy': accuracy}
        
        logger.info(f"🎯 Modelo entrenado - Precisión: {accuracy:.4f}")
        return model, accuracy, evaluation
        
    except Exception as e:
        logger.error(f"❌ Error en el pipeline de entrenamiento avanzado: {e}")
        raise

def get_feature_importance_dict(model, features) -> Dict:
    """Obtiene la importancia de características en formato diccionario"""
    try:
        if hasattr(model, 'feature_importances_'):
            return dict(zip(features, model.feature_importances_.tolist()))
        else:
            return {feature: 0.0 for feature in features}
    except:
        return {}

def save_model(model, accuracy, le_risk, features, scaler=None, model_dir='models') -> Tuple[str, str]:
    """
    Guarda el modelo entrenado y metadata de manera robusta
    """
    try:
        trainer = ModelTrainer(model_dir)
        
        # Generar nombres de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_dir}/student_risk_model_{timestamp}.pkl"
        metadata_filename = f"{model_dir}/model_metadata_{timestamp}.json"
        
        # Preparar metadata
        metadata = {
            'timestamp': timestamp,
            'training_date': datetime.now().isoformat(),
            'accuracy': float(accuracy),
            'features': features,
            'feature_importance': get_feature_importance_dict(model, features),
            'classes': dict(zip(range(len(le_risk.classes_)), le_risk.classes_.tolist())),
            'model_type': type(model).__name__,
            'model_parameters': model.get_params(),
            'dataset_info': {
                'n_features': len(features),
                'classes_count': len(le_risk.classes_)
            }
        }
        
        # Guardar modelo
        model_data = {
            'model': model,
            'metadata': metadata,
            'label_encoder': le_risk,
            'scaler': scaler
        }
        
        joblib.dump(model_data, model_filename)
        
        # Guardar metadata en JSON legible
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Modelo guardado en: {model_filename}")
        logger.info(f"📝 Metadata guardada en: {metadata_filename}")
        
        return model_filename, metadata_filename
        
    except Exception as e:
        logger.error(f"❌ Error guardando modelo: {e}")
        raise

def save_model_with_versioning(model, accuracy, le_risk, features, scaler=None, 
                              metadata: Dict = None, model_dir='models') -> Dict[str, Any]:
    """
    Guarda el modelo con sistema de versionado mejorado
    """
    try:
        trainer = AdvancedModelTrainer(model_dir)
        
        # Generar versión semántica
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"1.0.0_{timestamp}"  # Podría leerse de un archivo de versión
        
        model_filename = f"{model_dir}/student_risk_model_v{version}.pkl"
        metadata_filename = f"{model_dir}/model_metadata_v{version}.json"
        
        # Metadata enriquecida
        enhanced_metadata = {
            'version': version,
            'timestamp': timestamp,
            'training_date': datetime.now().isoformat(),
            'accuracy': float(accuracy),
            'features': features,
            'feature_importance': get_feature_importance_dict(model, features),
            'classes': dict(zip(range(len(le_risk.classes_)), le_risk.classes_.tolist())),
            'model_type': type(model).__name__,
            'model_parameters': model.get_params(),
            'dataset_info': {
                'n_features': len(features),
                'classes_count': len(le_risk.classes_)
            },
            'performance_metrics': {
                'accuracy': float(accuracy),
                'cross_validation': getattr(model, 'cv_score', None)
            }
        }
        
        if metadata:
            enhanced_metadata.update(metadata)
        
        # Guardar modelo
        model_data = {
            'model': model,
            'metadata': enhanced_metadata,
            'label_encoder': le_risk,
            'scaler': scaler,
            'version': version
        }
        
        joblib.dump(model_data, model_filename)
        
        # Guardar metadata en JSON
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(enhanced_metadata, f, indent=2, ensure_ascii=False)
        
        # Actualizar último modelo
        latest_path = f"{model_dir}/student_risk_model_latest.pkl"
        joblib.dump(model_data, latest_path)
        
        logger.info(f"💾 Modelo guardado: {model_filename}")
        logger.info(f"🔗 Último modelo actualizado: {latest_path}")
        
        return {
            'model_path': model_filename,
            'metadata_path': metadata_filename,
            'version': version,
            'latest_path': latest_path
        }
        
    except Exception as e:
        logger.error(f"❌ Error guardando modelo: {e}")
        raise

def load_latest_model(model_dir='models') -> Optional[Any]:
    """
    Carga el modelo más reciente de manera robusta
    """
    try:
        if not os.path.exists(model_dir):
            logger.warning(f"📁 Directorio de modelos no encontrado: {model_dir}")
            return None
        
        # Buscar archivos de modelo
        model_files = [f for f in os.listdir(model_dir) 
                     if f.endswith('.pkl') and 'student_risk_model' in f]
        
        if not model_files:
            logger.warning("❌ No se encontraron modelos guardados")
            return None
        
        # Ordenar por fecha (más reciente primero)
        model_files.sort(reverse=True)
        latest_model_path = os.path.join(model_dir, model_files[0])
        
        logger.info(f"📦 Cargando modelo: {latest_model_path}")
        
        # Cargar modelo
        model_data = joblib.load(latest_model_path)
        
        # Validar estructura del modelo cargado
        if not isinstance(model_data, dict) or 'model' not in model_data:
            logger.error("❌ Estructura inválida en archivo de modelo")
            return None
        
        model = model_data['model']
        metadata = model_data.get('metadata', {})
        
        logger.info(f"✅ Modelo cargado - Precisión: {metadata.get('accuracy', 'N/A')}")
        logger.info(f"📊 Modelo info - Características: {len(metadata.get('features', []))}")
        
        return model_data
        
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")
        return None

def get_model_info(model_data: Dict) -> Dict[str, Any]:
    """Obtiene información del modelo cargado"""
    if model_data is None:
        return {'error': 'Modelo no cargado'}
    
    metadata = model_data.get('metadata', {})
    model = model_data.get('model')
    
    info = {
        'accuracy': metadata.get('accuracy', 'N/A'),
        'training_date': metadata.get('training_date', 'N/A'),
        'features_count': len(metadata.get('features', [])),
        'model_type': metadata.get('model_type', 'N/A'),
        'classes': metadata.get('classes', {}),
        'feature_importance': metadata.get('feature_importance', {})
    }
    
    if model is not None:
        info['is_fitted'] = hasattr(model, 'feature_importances_')
    
    return info

# Nueva función para explicabilidad del modelo
def explain_model_predictions(model, X_test, feature_names, top_features: int = 10):
    """Provee explicaciones detalladas de las predicciones del modelo"""
    try:
        import shap
        
        # Crear explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Resumen de importancia de características
        if isinstance(shap_values, list):
            shap_sum = np.abs(shap_values[0]).mean(axis=0)
        else:
            shap_sum = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': shap_sum
        }).sort_values('importance', ascending=False).head(top_features)
        
        # Visualización (opcional)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        
        return {
            'feature_importance': importance_df.to_dict('records'),
            'shap_values': shap_values,
            'explainer': explainer
        }
        
    except Exception as e:
        logger.error(f"⚠️ Error en explicabilidad: {e}")
        return None

if __name__ == "__main__":
    # Configurar logging para pruebas
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Añadir el directorio raíz al sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, project_root)
    
    try:
        from src.data.data_loader import load_student_data
        from src.data.preprocessing import preprocess_student_data
    except ImportError as e:
        logger.error(f"❌ Error de importación: {e}")
        sys.exit(1)
    
    # Pipeline completo de entrenamiento
    print("🚀 Ejecutando model_training.py - PRUEBAS")
    print("=" * 60)
    
    df = load_student_data()
    if df is not None:
        X, y, le_risk, scaler = preprocess_student_data(df)
        
        if all([X is not None, y is not None, le_risk is not None]):
            print("✅ Datos preprocesados correctamente")
            
            # Entrenar modelo avanzado
            model, accuracy, evaluation = train_advanced_risk_model(X, y)
            
            # Guardar modelo - CORREGIDO: ahora manejamos el diccionario retornado
            features = X.columns.tolist()
            save_result = save_model_with_versioning(model, accuracy, le_risk, features, scaler)
            model_path = save_result['model_path']
            metadata_path = save_result['metadata_path']
            
            print(f"📁 Modelo guardado en: {model_path}")
            print(f"📊 Precisión del modelo: {accuracy:.4f}")
            
            # Probar carga del modelo
            loaded_model_data = load_latest_model()
            if loaded_model_data:
                model_info = get_model_info(loaded_model_data)
                print(f"🔍 Información del modelo cargado:")
                print(f"   - Precisión: {model_info['accuracy']}")
                print(f"   - Características: {model_info['features_count']}")
                print(f"   - Tipo: {model_info['model_type']}")
        else:
            print("❌ Error en el preprocesamiento de datos")
    else:
        print("❌ No se pudieron cargar los datos")

# 🔧 Añadir estas funciones al final de tu model_training.py existente

def get_model_compatibility_info(model_data: Dict) -> Dict[str, Any]:
    """
    Obtiene información de compatibilidad del modelo para el sistema de feedback
    """
    if model_data is None:
        return {'error': 'Modelo no cargado', 'compatible': False}
    
    try:
        model = model_data.get('model')
        metadata = model_data.get('metadata', {})
        le_risk = model_data.get('label_encoder')
        scaler = model_data.get('scaler')
        
        compatibility_info = {
            'compatible': True,
            'model_type': type(model).__name__ if model else 'Unknown',
            'has_label_encoder': le_risk is not None,
            'has_scaler': scaler is not None,
            'features': metadata.get('features', []),
            'classes': list(le_risk.classes_) if le_risk else [],
            'feature_count': len(metadata.get('features', [])),
            'model_trained': hasattr(model, 'feature_importances_') if model else False
        }
        
        # Verificar características críticas para feedback
        required_features = [
            'tasa_asistencia', 'completacion_tareas', 'puntuacion_participacion',
            'promedio_calificaciones', 'actividades_extracurriculares', 'involucramiento_parental_codificado'
        ]
        
        available_features = metadata.get('features', [])
        missing_features = [f for f in required_features if f not in available_features]
        
        if missing_features:
            compatibility_info['compatible'] = False
            compatibility_info['missing_features'] = missing_features
            compatibility_info['warning'] = f'Faltan características: {missing_features}'
        
        return compatibility_info
        
    except Exception as e:
        return {'error': str(e), 'compatible': False}

def validate_feedback_compatibility(model_data: Dict, student_data: Dict) -> Tuple[bool, str]:
    """
    Valida que los datos de estudiante sean compatibles con el modelo para feedback
    """
    try:
        if model_data is None:
            return False, "Modelo no disponible"
        
        metadata = model_data.get('metadata', {})
        model_features = set(metadata.get('features', []))
        
        # Mapear características de entrada a características del modelo
        input_mapping = {
            'tasa_asistencia': 'tasa_asistencia',
            'completacion_tareas': 'completacion_tareas', 
            'puntuacion_participacion': 'puntuacion_participacion',
            'promedio_calificaciones': 'promedio_calificaciones',
            'actividades_extracurriculares': 'actividades_extracurriculares',
            'involucramiento_parental': 'involucramiento_parental_codificado'
        }
        
        # Verificar que tenemos todas las características requeridas
        required_inputs = set(input_mapping.keys())
        available_inputs = set(student_data.keys())
        missing_inputs = required_inputs - available_inputs
        
        if missing_inputs:
            return False, f"Faltan datos de entrada: {list(missing_inputs)}"
        
        # Verificar que las características mapeadas estén en el modelo
        mapped_features = set(input_mapping.values())
        missing_model_features = mapped_features - model_features
        
        if missing_model_features:
            return False, f"El modelo no tiene características esperadas: {list(missing_model_features)}"
        
        return True, "Compatible"
        
    except Exception as e:
        return False, f"Error en validación: {str(e)}"

def prepare_student_for_feedback(student_data: Dict, feature_names: List[str]) -> Dict:
    """
    Prepara datos de estudiante para el sistema de feedback
    """
    try:
        # Mapear engagement parental a formato numérico
        engagement_mapping = {
            'Bajo': 0, 'Medio': 1, 'Alto': 2,
            'Faible': 0, 'Moyenne': 1, 'Élevée': 2
        }
        
        prepared_data = {
            'tasa_asistencia': float(student_data['tasa_asistencia']),
            'completacion_tareas': float(student_data['completacion_tareas']),
            'puntuacion_participacion': float(student_data['puntuacion_participacion']),
            'promedio_calificaciones': float(student_data['promedio_calificaciones']),
            'actividades_extracurriculares': int(student_data['actividades_extracurriculares']),
            'involucramiento_parental_codificado': engagement_mapping.get(
                student_data['involucramiento_parental'], 1  # Default to 'Medio'
            )
        }
        
        return prepared_data
        
    except Exception as e:
        logger.error(f"Error preparando datos para feedback: {e}")
        raise
