"""
Script para verificar que el proyecto esté configurado correctamente
"""
import os
import sys
import importlib

def check_directory_structure():
    """Verifica que la estructura de directorios sea correcta"""
    print("📁 Verificando estructura de directorios...")
    
    required_dirs = [
        'data', 
        'logs',
        'models',
        'models/versions',
        'src',
        'src/data',
        'src/ml',
        '.streamlit'
    ]
    
    required_files = [
        'streamlit_app.py',
        'data/student_risk_indicators_v2 (1).csv',
        'src/data/__init__.py',
        'src/data/data_loader.py',
        'src/data/preprocessing.py',
        'src/ml/__init__.py',
        'src/ml/model_training.py',
        'src/ml/recommendation_system.py',
        'src/ml/feedback_system.py',
        '.streamlit/config.toml',
        'requirements.txt',
        'README.md'
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directorio: {dir_path}")
        else:
            print(f"❌ FALTA Directorio: {dir_path}")
            all_good = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ Archivo: {file_path}")
        else:
            print(f"❌ FALTA Archivo: {file_path}")
            all_good = False
    
    return all_good

def check_imports():
    """Verifica que todos los imports funcionen"""
    print("\n🔍 Verificando imports...")
    
    modules_to_check = [
        'streamlit',
        'pandas',
        'numpy', 
        'sklearn',
        'plotly',
        'matplotlib',
        'shap',
        'joblib',
        'jinja2'
    ]
    
    custom_modules = [
        'src.data.data_loader',
        'src.data.preprocessing',
        'src.ml.model_training',
        'src.ml.recommendation_system',
        'src.ml.feedback_system'
    ]
    
    all_good = True
    
    # Verificar librerías externas
    for module in modules_to_check:
        try:
            importlib.import_module(module)
            print(f"✅ Librería: {module}")
        except ImportError as e:
            print(f"❌ FALTA Librería: {module} - {e}")
            all_good = False
    
    # Verificar módulos custom (con manejo de paths)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    for module in custom_modules:
        try:
            importlib.import_module(module)
            print(f"✅ Módulo: {module}")
        except ImportError as e:
            print(f"❌ ERROR Módulo: {module} - {e}")
            all_good = False
    
    return all_good

def check_data_loading():
    """Verifica que los datos se carguen correctamente"""
    print("\n📊 Verificando carga de datos...")
    
    try:
        # Asegurar que el path esté configurado
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        from src.data.data_loader import load_student_data
        
        df = load_student_data()
        if df is not None:
            print(f"✅ Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
            print(f"✅ Columnas: {list(df.columns)}")
            return True
        else:
            print("❌ No se pudieron cargar los datos")
            return False
            
    except Exception as e:
        print(f"❌ Error en carga de datos: {e}")
        return False

def check_streamlit_config():
    """Verifica la configuración de Streamlit"""
    print("\n⚙️ Verificando configuración de Streamlit...")
    
    config_file = '.streamlit/config.toml'
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print("✅ Configuración de Streamlit cargada correctamente")
                return True
        except Exception as e:
            print(f"❌ Error al leer config.toml: {e}")
            return False
    else:
        print("❌ No se encontró el archivo config.toml")
        return False

if __name__ == "__main__":
    print("🚀 INICIANDO VERIFICACIÓN DEL PROYECTO")
    print("=" * 50)
    
    dirs_ok = check_directory_structure()
    imports_ok = check_imports() 
    data_ok = check_data_loading()
    streamlit_ok = check_streamlit_config()
    
    print("\n" + "=" * 50)
    print("📋 RESUMEN DE VERIFICACIÓN:")
    print(f"Estructura de directorios: {'✅ OK' if dirs_ok else '❌ PROBLEMAS'}")
    print(f"Imports: {'✅ OK' if imports_ok else '❌ PROBLEMAS'}")
    print(f"Carga de datos: {'✅ OK' if data_ok else '❌ PROBLEMAS'}")
    print(f"Configuración Streamlit: {'✅ OK' if streamlit_ok else '❌ PROBLEMAS'}")
    
    if all([dirs_ok, imports_ok, data_ok, streamlit_ok]):
        print("\n🎉 ¡PROYECTO CONFIGURADO CORRECTAMENTE!")
        print("Puedes ejecutar: streamlit run app/streamlit_app.py")
    else:
        print("\n⚠️  Hay problemas que necesitan atención.")
        print("Revisa los mensajes de error arriba.")