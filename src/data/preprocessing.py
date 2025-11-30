import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
import sys
import logging
from typing import Tuple, Optional, Any

# Configurar logging
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Clase para preprocesamiento robusto de datos educativos"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder_risk = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
    def validate_input_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Valida que los datos de entrada sean adecuados para el preprocesamiento"""
        if df is None or df.empty:
            return False, "DataFrame vacío o None"
        
        required_columns = [
            'tasa_asistencia', 'completacion_tareas', 'puntuacion_participacion',
            'promedio_calificaciones', 'actividades_extracurriculares', 'involucramiento_parental', 'nivel_riesgo'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Columnas faltantes: {missing_columns}"
        
        return True, "OK"
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maneja valores faltantes de manera robusta"""
        df_clean = df.copy()
        
        # Estrategias específicas por columna
        imputation_strategies = {
            'tasa_asistencia': 'median',
            'completacion_tareas': 'median', 
            'puntuacion_participacion': 'median',
            'promedio_calificaciones': 'median',
            'actividades_extracurriculares': 'most_frequent',
            'involucramiento_parental': 'most_frequent',
            'nivel_riesgo': 'most_frequent'
        }
        
        for column, strategy in imputation_strategies.items():
            if column in df_clean.columns:
                if df_clean[column].isnull().any():
                    if strategy == 'median':
                        fill_value = df_clean[column].median()
                    elif strategy == 'mean':
                        fill_value = df_clean[column].mean()
                    else:  # most_frequent
                        fill_value = df_clean[column].mode()[0] if not df_clean[column].mode().empty else 'Bajo'  # Cambiado de 'Faible' a 'Bajo'
                    
                    df_clean[column].fillna(fill_value, inplace=True)
                    logger.info(f"🔧 Imputados {df_clean[column].isnull().sum()} valores en {column} con {strategy}")
        
        return df_clean
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Codifica variables categóricas de manera segura - CORREGIDO A ESPAÑOL"""
        df_encoded = df.copy()
        
        # Mapeo para involucramiento_parental - CORREGIDO A ESPAÑOL
        engagement_mapping = {'Bajo': 0, 'Medio': 1, 'Alto': 2}  # Cambiado de francés a español
        df_encoded['involucramiento_parental_codificado'] = df_encoded['involucramiento_parental'].map(engagement_mapping)
        
        # Manejar valores no mapeados
        if df_encoded['involucramiento_parental_codificado'].isnull().any():
            logger.warning("⚠️ Valores no mapeados en involucramiento_parental, usando valor por defecto (Bajo=0)")
            df_encoded['involucramiento_parental_codificado'].fillna(0, inplace=True)
        
        # Codificar nivel_riesgo
        try:
            df_encoded['nivel_riesgo_codificado'] = self.label_encoder_risk.fit_transform(df_encoded['nivel_riesgo'])
            logger.info(f"🎯 Niveles de riesgo codificados: {dict(enumerate(self.label_encoder_risk.classes_))}")
        except Exception as e:
            logger.error(f"Error codificando nivel_riesgo: {e}")
            # Fallback: mapeo manual - CORREGIDO A ESPAÑOL
            risk_mapping = {'Bajo': 0, 'Medio': 1, 'Alto': 2}  # Cambiado de francés a español
            df_encoded['nivel_riesgo_codificado'] = df_encoded['nivel_riesgo'].map(risk_mapping)
            
            # Manejar valores no mapeados
            if df_encoded['nivel_riesgo_codificado'].isnull().any():
                logger.warning("⚠️ Valores no mapeados en nivel_riesgo, usando valor por defecto (Bajo=0)")
                df_encoded['nivel_riesgo_codificado'].fillna(0, inplace=True)
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, features: list) -> Tuple[pd.DataFrame, StandardScaler]:
        """Escala las características numéricas"""
        try:
            # Verificar que todas las features existan
            available_features = [f for f in features if f in df.columns]
            missing_features = [f for f in features if f not in df.columns]
            
            if missing_features:
                logger.warning(f"⚠️ Características faltantes para escalado: {missing_features}")
            
            if available_features:
                # Verificar que no haya valores infinitos o NaN
                for feature in available_features:
                    if df[feature].isnull().any():
                        df[feature].fillna(df[feature].median(), inplace=True)
                    if np.isinf(df[feature]).any():
                        df[feature].replace([np.inf, -np.inf], df[feature].median(), inplace=True)
                
                scaled_values = self.scaler.fit_transform(df[available_features])
                df_scaled = df.copy()
                df_scaled[available_features] = scaled_values
                logger.info(f"📏 Características escaladas: {available_features}")
                return df_scaled, self.scaler
            else:
                logger.warning("⚠️ No hay características disponibles para escalar")
                return df, self.scaler
                
        except Exception as e:
            logger.error(f"❌ Error en escalado de características: {e}")
            return df, self.scaler

def preprocess_student_data(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[LabelEncoder], Optional[StandardScaler]]:
    """
    Preprocesa los datos de estudiantes para el modelo de manera robusta
    """
    try:
        preprocessor = DataPreprocessor()
        
        # Validar datos de entrada
        is_valid, message = preprocessor.validate_input_data(df)
        if not is_valid:
            logger.error(f"❌ Datos de entrada inválidos: {message}")
            return None, None, None, None
        
        logger.info("🔄 Iniciando preprocesamiento de datos...")
        
        # 1. Manejar valores faltantes
        df_clean = preprocessor.handle_missing_values(df)
        logger.info(f"✅ Valores faltantes manejados. Shape: {df_clean.shape}")
        
        # 2. Codificar variables categóricas
        df_encoded = preprocessor.encode_categorical_features(df_clean)
        logger.info("✅ Variables categóricas codificadas")
        
        # 3. Seleccionar características relevantes
        features = [
            'tasa_asistencia', 
            'completacion_tareas', 
            'puntuacion_participacion', 
            'promedio_calificaciones',
            'actividades_extracurriculares',
            'involucramiento_parental_codificado'  # Usar la columna codificada
        ]
        
        # Verificar que todas las features estén disponibles
        available_features = [f for f in features if f in df_encoded.columns]
        if len(available_features) != len(features):
            missing = set(features) - set(available_features)
            logger.warning(f"⚠️ Características no disponibles: {missing}")
            
            # Si falta la columna codificada, intentar crear una básica
            if 'involucramiento_parental_codificado' not in available_features and 'involucramiento_parental' in df_encoded.columns:
                logger.info("🔧 Creando codificación básica para involucramiento_parental")
                basic_mapping = {'Bajo': 0, 'Medio': 1, 'Alto': 2}
                df_encoded['involucramiento_parental_codificado'] = df_encoded['involucramiento_parental'].map(basic_mapping).fillna(0)
                available_features.append('involucramiento_parental_codificado')
        
        X = df_encoded[available_features]
        y = df_encoded['nivel_riesgo_codificado']
        
        # 4. Escalar características numéricas
        X_scaled, scaler = preprocessor.scale_features(X, available_features)
        
        logger.info("✅ Preprocesamiento completado exitosamente")
        logger.info(f"📊 Dataset final - X: {X_scaled.shape}, y: {y.shape}")
        logger.info(f"🎯 Distribución de clases: {pd.Series(y).value_counts().to_dict()}")
        logger.info(f"🔧 Características utilizadas: {available_features}")
        
        preprocessor.is_fitted = True
        return X_scaled, y, preprocessor.label_encoder_risk, scaler
        
    except Exception as e:
        logger.error(f"❌ Error en preprocesamiento: {e}")
        return None, None, None, None

def prepare_new_student_data(student_data: dict, scaler: StandardScaler, features: list) -> np.ndarray:
    """
    Prepara datos de un nuevo estudiante para la predicción de manera robusta - CORREGIDO A ESPAÑOL
    """
    try:
        # Validar datos de entrada
        if not isinstance(student_data, dict):
            raise ValueError("student_data debe ser un diccionario")
        
        if scaler is None:
            raise ValueError("Scaler no puede ser None")
        
        # Crear DataFrame con los datos del estudiante
        student_dict = {
            'tasa_asistencia': float(student_data.get('tasa_asistencia', 0)),
            'completacion_tareas': float(student_data.get('completacion_tareas', 0)),
            'puntuacion_participacion': float(student_data.get('puntuacion_participacion', 0)),
            'promedio_calificaciones': float(student_data.get('promedio_calificaciones', 0)),
            'actividades_extracurriculares': int(student_data.get('actividades_extracurriculares', 0)),
        }
        
        # Mapear involucramiento parental de manera segura - CORREGIDO A ESPAÑOL
        engagement_mapping = {'Bajo': 0, 'Medio': 1, 'Alto': 2}  # Cambiado de francés a español
        involucramiento_parental = student_data.get('involucramiento_parental', 'Bajo')
        student_dict['involucramiento_parental_codificado'] = engagement_mapping.get(involucramiento_parental, 0)
        
        # Crear DataFrame
        df_student = pd.DataFrame([student_dict])
        
        # Seleccionar y escalar características disponibles
        available_features = [f for f in features if f in df_student.columns]
        if not available_features:
            raise ValueError("No hay características disponibles para la predicción")
        
        X_new = df_student[available_features]
        X_new_scaled = scaler.transform(X_new)
        
        logger.info(f"✅ Datos de estudiante preparados. Características: {available_features}")
        return X_new_scaled
        
    except Exception as e:
        logger.error(f"❌ Error preparando datos de nuevo estudiante: {e}")
        raise

def get_preprocessing_summary(X, y, le_risk, scaler) -> dict:
    """Provee un resumen del preprocesamiento realizado"""
    summary = {
        'cantidad_caracteristicas': X.shape[1] if X is not None else 0,
        'cantidad_muestras': len(y) if y is not None else 0,
        'clases_codificadas': dict(enumerate(le_risk.classes_)) if le_risk else {},
        'escalador_ajustado': hasattr(scaler, 'mean_') if scaler else False,
        'nombres_caracteristicas': list(X.columns) if X is not None else []
    }
    return summary

if __name__ == "__main__":
    # Configurar logging para pruebas
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Añadir el directorio raíz al sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, project_root)
    
    # Importar desde el módulo corregido
    from src.data.data_loader import load_student_data
    
    # Prueba de preprocesamiento
    print("🚀 Ejecutando preprocessing.py - PRUEBAS")
    print("=" * 60)
    
    df = load_student_data()
    if df is not None:
        X, y, le_risk, scaler = preprocess_student_data(df)
        
        if all([X is not None, y is not None, le_risk is not None, scaler is not None]):
            summary = get_preprocessing_summary(X, y, le_risk, scaler)
            print(f"\n✅ Preprocesamiento exitoso:")
            print(f"   Características: {summary['cantidad_caracteristicas']}")
            print(f"   Muestras: {summary['cantidad_muestras']}")
            print(f"   Clases: {summary['clases_codificadas']}")
            print(f"   Escalador ajustado: {summary['escalador_ajustado']}")
        else:
            print("❌ Error en el preprocesamiento")
