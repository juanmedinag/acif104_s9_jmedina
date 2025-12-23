# ACIF104 – Sistema de Mantenimiento Predictivo

Este repositorio contiene el desarrollo del proyecto final del curso ACIF104 – Aprendizaje de Máquinas,
correspondiente a un sistema de mantenimiento predictivo orientado al sector público.

El sistema utiliza técnicas de machine learning supervisado y deep learning, integradas en una
arquitectura con backend y frontend desacoplados, incorporando criterios de explicabilidad
y **monitoreo del desempeño del modelo**.

---

## 1. Descripción general

El objetivo del sistema es **estimar la probabilidad de falla de equipos críticos** a partir de
variables operacionales históricas, permitiendo anticipar intervenciones de mantenimiento
preventivo y apoyar la toma de decisiones técnicas.

El proyecto utiliza un dataset sintético basado en **AI4I 2020 Predictive Maintenance**, adaptado
conceptualmente al contexto de una institución del sector público con restricciones de recursos
computacionales y altos requerimientos de trazabilidad y explicabilidad.

---

## 2. Arquitectura del sistema

El sistema se compone de los siguientes módulos:

- **Preprocesamiento de datos**
  - Limpieza y estandarización de variables numéricas.
  - Manejo de clases desbalanceadas.
- **Modelos de aprendizaje automático**
  - *Gradient Boosting* como modelo supervisado principal para predicción de fallas.
  - *Autoencoder profundo* como modelo complementario para detección de anomalías.
- **Backend**
  - API REST implementada con **FastAPI**.
  - Endpoints para predicción de fallas y consulta de métricas del modelo.
- **Frontend**
  - Dashboard web desarrollado en **Streamlit**.
  - Visualización de probabilidad de falla y resultados explicables.
- **Monitoreo**
  - Detección de drift de datos.
  - Registro de métricas de desempeño del modelo.
  - Definición de alertas para reentrenamiento.

---

## 3. Estructura del repositorio

.
├── backend/              # API REST (FastAPI)
├── frontend/             # Dashboard web (Streamlit)
├── notebooks/            # EDA, entrenamiento y análisis SHAP
├── data/                 # Dataset o referencias
├── models/               # Modelos entrenados serializados
├── monitoring/           # Módulos de monitoreo (drift)
├── requirements.txt
└── README.md

---

## 4. Explicabilidad del modelo

La explicabilidad del sistema se aborda mediante el uso de **SHAP**, permitiendo:

- Interpretación global de la importancia de las variables.
- Análisis local por predicción individual.
- Soporte a la toma de decisiones por personal no especializado.

Los gráficos SHAP se generan en los notebooks de análisis y se integran al dashboard frontend
para su visualización.

---

## 5. Monitoreo del modelo

El sistema considera un módulo de monitoreo **conceptual y extensible**, que incluye:

- **Detección de drift de datos** mediante comparación de distribuciones (Population Stability Index – PSI).
- **Registro histórico de métricas de desempeño** del modelo (precision, recall, F1-score, AUC).
- **Alertas automáticas** cuando el desempeño cae bajo umbrales definidos, habilitando procesos
  de reentrenamiento.

Estas funcionalidades están diseñadas para su implementación progresiva en un entorno de
producción institucional.

---

## 6. Instalación y ejecución

### 6.1 Crear entorno virtual

python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

### 6.2 Instalar dependencias

pip install -r requirements.txt

### 6.3 Ejecutar backend

uvicorn backend.main:app --reload

### 6.4 Ejecutar frontend

streamlit run frontend/app.py

---

## 7. Requisitos técnicos

- Python 3.9 o superior
- scikit-learn
- pandas
- numpy
- FastAPI
- Streamlit
- SHAP
- imbalanced-learn
- joblib

---

## 8. Contexto académico

Este repositorio corresponde al **proyecto final del curso ACIF104 – Aprendizaje de Máquinas**
y debe ser leído en conjunto con el **informe técnico entregado en la Semana 12**, donde se
documenta en detalle la metodología aplicada, la evaluación de modelos y los resultados obtenidos.
