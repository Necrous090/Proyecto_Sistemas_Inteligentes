# Sistema Inteligente de Recomendación Educativa

## 📋 Descripción
El Sistema Inteligente de Recomendación Educativa es una plataforma avanzada que utiliza Inteligencia Artificial y Machine Learning para analizar indicadores de rendimiento académico, identificar estudiantes en riesgo y generar recomendaciones personalizadas con justificaciones basadas en datos.

Permite a docentes, orientadores y administradores tomar decisiones proactivas, optimizar recursos educativos y aplicar intervenciones oportunas.

## 🚀Características Principales
- 📊 Análisis predictivo del riesgo académico
- 🎯 Recomendaciones personalizadas por perfil
- 🧠 Justificación basada en IA mediante SHAP
- 🔄 Aprendizaje continuo mediante sistema de feedback
- 📈 Visualizaciones interactivas
- 📱 Interfaz intuitiva con Streamlit
- 🔐 Manejo seguro y responsable de datos

## 🎯 Objetivo
Desarrollar una herramienta analítica versátil que identifique patrones ocultos, prediga necesidades futuras y proponga soluciones concretas basadas en Inteligencia Artificial.

## 📁 Estructura del Proyecto
```bash
ProyectoFinalSI/
├── streamlit_app.py
├── data/
│   └── student_risk_indicators_v2 (1).csv
├── src/
│   ├── data/
│   │   └── data_loader.py
│   ├── preprocessing.py
│   ├── ml/
│   │   └── model_training.py
│   ├── recommendation_system.py
│   └── feedback_system.py
├── models/
├── feedback_data/
├── logs/
├── .streamlit/
│   └── config.toml
├── requirements.txt
└── README.md```

---

## 🚀 Cómo ejecutar
1. Instalar dependencias: `pip install -r requirements.txt`
2. Ejecutar la aplicación: `streamlit run app/streamlit_app.py`

## ▶️ Ejecución de la Aplicación
    `streamlit run streamlit_app.py`

La aplicación se abrirá en:
👉 `http://localhost:8501`

## 🌐 Despliegue en Streamlit Cloud
1. Crear una cuenta en Streamlit Cloud
2. Conectar el repositorio
3. Verificar requirements.txt
4. Seleccionar el archivo principal:
5. streamlit_app.py
6. Desplegar y compartir





