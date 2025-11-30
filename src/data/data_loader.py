import pandas as pd
import os
import sys
import logging
import numpy as np
from typing import Optional, Dict, Any, List
import requests
from io import StringIO

# Configurar logging
logger = logging.getLogger(__name__)

class DataLoader:
    """Clase mejorada para carga y validación de datos de estudiantes"""
    
    def __init__(self):
        self.possible_paths = [
            'student_risk_indicators_v2 (1).csv',
            'data/student_risk_indicators_v2 (1).csv',
            '../data/student_risk_indicators_v2 (1).csv',
            './data/student_risk_indicators_v2 (1).csv',
            'app/data/student_risk_indicators_v2 (1).csv',
            'data/student_data.csv',
            'dataset.csv',
            'students.csv'
        ]
        
        # Cache para datos cargados
        self._cached_data = None
        self._cached_file_path = None
    
    def find_data_file(self) -> Optional[str]:
        """Busca el archivo de datos en múltiples ubicaciones con cache"""
        if hasattr(self, '_cached_file_path') and self._cached_file_path:
            return self._cached_file_path
            
        for path in self.possible_paths:
            if os.path.exists(path):
                logger.info(f"📁 Archivo encontrado en: {path}")
                self._cached_file_path = path
                return path
        
        # Búsqueda recursiva mejorada
        search_patterns = ['*student*', '*risk*', '*data*']
        for pattern in search_patterns:
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if (pattern.replace('*', '').lower() in file.lower() and 
                        file.endswith('.csv')):
                        found_path = os.path.join(root, file)
                        logger.info(f"📁 Archivo encontrado (búsqueda recursiva): {found_path}")
                        self._cached_file_path = found_path
                        return found_path
        
        return None

    def create_sample_data(self) -> pd.DataFrame:
        """Crea datos de ejemplo si no se encuentra el archivo"""
        logger.warning("📝 Creando datos de ejemplo para demo...")
        
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = {
            'ID': [f'ID_{i:04d}' for i in range(1, n_samples + 1)],
            'tasa_asistencia': np.random.normal(85, 15, n_samples).clip(0, 100),
            'completacion_tareas': np.random.normal(75, 20, n_samples).clip(0, 100),
            'puntuacion_participacion': np.random.normal(6.5, 2, n_samples).clip(1, 10),
            'promedio_calificaciones': np.random.normal(14, 3, n_samples).clip(0, 20),
            'actividades_extracurriculares': np.random.randint(0, 6, n_samples),
            'involucramiento_parental': np.random.choice(['Faible', 'Moyenne', 'Élevée'], n_samples, p=[0.3, 0.5, 0.2])
        }
        
        # Calcular riesgo basado en reglas simples
        def calculate_risk(row):
            score = 0
            if row['tasa_asistencia'] < 70: score += 2
            if row['completacion_tareas'] < 60: score += 2
            if row['puntuacion_participacion'] < 4: score += 1
            if row['promedio_calificaciones'] < 10: score += 2
            if row['involucramiento_parental'] == 'Faible': score += 1
            
            if score >= 4: return 'Élevé'
            elif score >= 2: return 'Moyen'
            else: return 'Faible'
        
        df = pd.DataFrame(sample_data)
        df['nivel_riesgo'] = df.apply(calculate_risk, axis=1)
        
        logger.info("✅ Datos de ejemplo creados exitosamente")
        return df

def load_student_data(file_path: Optional[str] = None, use_cache: bool = True) -> Optional[pd.DataFrame]:
    """
    Carga los datos de estudiantes con mejoras significativas
    """
    try:
        loader = DataLoader()
        
        # Usar cache si está disponible
        if use_cache and loader._cached_data is not None:
            logger.info("📦 Usando datos en caché")
            return loader._cached_data.copy()
        
        # Si no se especifica ruta, buscar en diferentes ubicaciones
        if file_path is None:
            file_path = loader.find_data_file()
        
        if file_path is None or not os.path.exists(file_path):
            logger.warning("❌ No se encontró archivo de datos, creando datos de ejemplo")
            df = loader.create_sample_data()
        else:
            logger.info(f"🔍 Cargando datos desde: {file_path}")
            
            # Detectar encoding automáticamente
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='latin-1')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
            
            # Validaciones mejoradas
            if df.empty:
                raise ValueError("El archivo CSV está vacío")
            
            # Mapear nombres de columnas antiguos a nuevos si es necesario
            column_mapping = {
                'attendance_rate': 'tasa_asistencia',
                'homework_completion': 'completacion_tareas', 
                'participation_score': 'puntuacion_participacion',
                'average_grades': 'promedio_calificaciones',
                'extracurricular_activities': 'actividades_extracurriculares',
                'parental_engagement': 'involucramiento_parental',
                'risk_level': 'nivel_riesgo'
            }
            
            # Renombrar columnas si existen las antiguas
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df = df.rename(columns={old_col: new_col})
        
        # Validaciones de columnas mejoradas
        required_columns = ['tasa_asistencia', 'completacion_tareas', 'puntuacion_participacion', 
                          'promedio_calificaciones', 'actividades_extracurriculares', 'involucramiento_parental', 'nivel_riesgo']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"⚠️ Columnas faltantes: {missing_columns}")
            # Intentar encontrar columnas similares
            available_columns = list(df.columns)
            column_mapping = {}
            
            for missing in missing_columns:
                for available in available_columns:
                    if missing.lower() in available.lower() or available.lower() in missing.lower():
                        column_mapping[missing] = available
                        break
            
            if column_mapping:
                logger.info(f"🔍 Mapeo de columnas encontrado: {column_mapping}")
                df = df.rename(columns=column_mapping)
        
        # Cachear datos
        loader._cached_data = df.copy()
        
        logger.info(f"✅ Datos cargados exitosamente. Shape: {df.shape}")
        logger.info(f"📊 Columnas: {list(df.columns)}")
        logger.info(f"🎯 Distribución de riesgo: {df['nivel_riesgo'].value_counts().to_dict()}")
        
        return df
    
    except Exception as e:
        logger.error(f"❌ Error inesperado al cargar datos: {e}")
        # Crear datos de ejemplo como fallback
        return DataLoader().create_sample_data()

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Obtiene un resumen detallado de los datos
    """
    if df is None or df.empty:
        return {"error": "No hay datos disponibles"}
    
    try:
        summary = {
            'total_estudiantes': len(df),
            'distribucion_riesgo': df['nivel_riesgo'].value_counts().to_dict(),
            'columnas': list(df.columns),
            'valores_faltantes': df.isnull().sum().to_dict(),
            'tipos_datos': df.dtypes.astype(str).to_dict(),
            'estadisticas_numericas': {},
            'estadisticas_categoricas': {}
        }
        
        # Estadísticas para columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary['estadisticas_numericas'][col] = {
                'media': df[col].mean(),
                'desviacion_estandar': df[col].std(),
                'minimo': df[col].min(),
                'maximo': df[col].max(),
                'mediana': df[col].median()
            }
        
        # Estadísticas para columnas categóricas
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            summary['estadisticas_categoricas'][col] = {
                'valores_unicos': df[col].nunique(),
                'valor_mas_comun': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'conteo_valor_mas_comun': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
            }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generando resumen de datos: {e}")
        return {"error": f"Error generando resumen: {e}"}

def validate_student_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Valida la integridad del DataFrame de estudiantes
    """
    validation_result = {
        'es_valido': True,
        'errores': [],
        'advertencias': []
    }
    
    if df is None:
        validation_result['es_valido'] = False
        validation_result['errores'].append("DataFrame es None")
        return validation_result
    
    # Verificar columnas requeridas
    required_columns = ['tasa_asistencia', 'completacion_tareas', 'puntuacion_participacion', 
                      'promedio_calificaciones', 'actividades_extracurriculares', 'involucramiento_parental', 'nivel_riesgo']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation_result['es_valido'] = False
        validation_result['errores'].append(f"Columnas faltantes: {missing_columns}")
    
    # Verificar valores nulos
    null_counts = df[required_columns].isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        validation_result['advertencias'].append(f"Columnas con valores nulos: {columns_with_nulls.to_dict()}")
    
    # Validar rangos de valores
    value_checks = [
        ('tasa_asistencia', 0, 100),
        ('completacion_tareas', 0, 100),
        ('puntuacion_participacion', 0, 10),
        ('promedio_calificaciones', 0, 20),
        ('actividades_extracurriculares', 0, 10)
    ]
    
    for col, min_val, max_val in value_checks:
        if col in df.columns:
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            if not out_of_range.empty:
                validation_result['advertencias'].append(f"Valores fuera de rango en {col}: {len(out_of_range)} registros")
    
    return validation_result

# Nueva función para análisis de calidad de datos
def analyze_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Análisis completo de calidad de datos
    """
    analysis = {
        'completitud': {},
        'consistencia': {},
        'validez': {},
        'anomalias': {}
    }
    
    # Completitud
    analysis['completitud']['total_faltantes'] = df.isnull().sum().sum()
    analysis['completitud']['faltantes_por_columna'] = df.isnull().sum().to_dict()
    analysis['completitud']['tasa_completitud'] = 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
    
    # Consistencia
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    analysis['consistencia']['estadisticas_numericas'] = df[numeric_cols].describe().to_dict()
    
    # Validez
    validity_checks = {
        'tasa_asistencia': (0, 100),
        'completacion_tareas': (0, 100),
        'puntuacion_participacion': (0, 10),
        'promedio_calificaciones': (0, 20)
    }
    
    for col, (min_val, max_val) in validity_checks.items():
        if col in df.columns:
            invalid_count = ((df[col] < min_val) | (df[col] > max_val)).sum()
            analysis['validez'][col] = {
                'conteo_invalidos': invalid_count,
                'porcentaje_invalido': invalid_count / len(df) * 100
            }
    
    # Anomalías
    from scipy import stats
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        anomalies = (z_scores > 3).sum()
        analysis['anomalias'][col] = anomalies
    
    return analysis

if __name__ == "__main__":
    # Configurar logging básico para pruebas
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Prueba de carga de datos
    print("🚀 Ejecutando data_loader.py - PRUEBAS")
    print("=" * 60)
    
    df = load_student_data()
    if df is not None:
        summary = get_data_summary(df)
        validation = validate_student_dataframe(df)
        quality = analyze_data_quality(df)
        
        print("\n📊 RESUMEN DE DATOS:")
        print(f"Total de estudiantes: {summary['total_estudiantes']}")
        print(f"Distribución de riesgo: {summary['distribucion_riesgo']}")
        
        print("\n🔍 VALIDACIÓN:")
        print(f"Válido: {validation['es_valido']}")
        if validation['errores']:
            print(f"Errores: {validation['errores']}")
        if validation['advertencias']:
            print(f"Advertencias: {validation['advertencias']}")
            
        print("\n📈 CALIDAD DE DATOS:")
        print(f"Tasa de completitud: {quality['completitud']['tasa_completitud']:.2%}")
        print(f"Valores faltantes totales: {quality['completitud']['total_faltantes']}")
    else:
        print("❌ No se pudieron cargar los datos")