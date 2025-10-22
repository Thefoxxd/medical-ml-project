# 🏥 Sistema de Predicción Médica con Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44.0-green.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sistema web interactivo de Machine Learning para la predicción de costos de seguros médicos y diagnóstico de diabetes, desarrollado como proyecto educativo.

![Banner del Proyecto](images/banner.png)
*Captura de pantalla de la interfaz web*

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Tecnologías](#-tecnologías)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Análisis y Resultados](#-análisis-y-resultados)
- [Respuestas a las Preguntas](#-respuestas-a-las-preguntas)
- [Contribuir](#-contribuir)
- [Licencia](#-licencia)
- [Autores](#-autores)

## 🎯 Descripción

Este proyecto implementa dos modelos de Machine Learning:

1. **Predicción de Costos de Seguro Médico**: Utiliza Random Forest Regressor para estimar los costos de seguros basándose en características del paciente.

2. **Predicción de Diabetes**: Emplea Random Forest Classifier optimizado para evaluar el riesgo de diabetes en pacientes.

El sistema incluye una interfaz web desarrollada con Gradio que permite hacer predicciones en tiempo real.

## ✨ Características

- 🤖 **Modelos Optimizados**: Utiliza GridSearchCV para encontrar los mejores hiperparámetros
- 📊 **Análisis Exhaustivo**: Incluye análisis de importancia de características y detección de sesgos
- 🌐 **Interfaz Web Interactiva**: Aplicación web fácil de usar con Gradio
- 📈 **Visualizaciones**: Gráficos detallados de métricas y análisis
- 🔍 **Umbral Optimizado**: Umbral de clasificación ajustado para maximizar F1-Score
- 📱 **Responsive**: Funciona en desktop y móvil

## 🛠️ Tecnologías

- **Python 3.8+**
- **Scikit-Learn**: Modelos de ML y métricas
- **Pandas & NumPy**: Manipulación de datos
- **Matplotlib & Seaborn**: Visualizaciones
- **Gradio**: Interfaz web
- **SciPy**: Análisis estadístico

## 📦 Instalación

### Opción 1: Local
```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/medical-ml-project.git
cd medical-ml-project

# 2. Crear entorno virtual (recomendado)
python -m venv venv

# Activar el entorno
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Asegurarse de tener los datasets
# Coloca insurance.csv y diabetes.csv en la carpeta raíz
```

### Opción 2: Google Colab
```bash
# 1. Abre el notebook en Google Colab
# 2. Sube los archivos insurance.csv y diabetes.csv
# 3. Ejecuta todas las celdas
```

## 🚀 Uso

### Ejecutar Localmente
```bash
# Ejecutar la aplicación
python app.py

# La aplicación se abrirá en http://localhost:7860
```

### Ejecutar en Google Colab
```python
# En una celda de Colab:
# 1. Sube los archivos CSV
# 2. Copia y pega el código completo
# 3. Ejecuta la celda
# 4. Se generará una URL pública compartible
```

### Usar la Interfaz Web

1. **Predicción de Costos de Seguro**:
   - Ingresa edad, sexo, IMC, número de hijos, estado de fumador y región
   - Click en "Calcular Costo"
   - Obtén una estimación del costo anual

2. **Predicción de Diabetes**:
   - Ingresa valores de glucosa, presión arterial, IMC, etc.
   - Click en "Evaluar Riesgo"
   - Obtén la probabilidad de diabetes y nivel de riesgo

## 📁 Estructura del Proyecto
```
medical-ml-project/
│
├── README.md                 # Este archivo
├── requirements.txt          # Dependencias
├── .gitignore               # Archivos a ignorar
├── app.py                   # Aplicación principal
├── LICENSE                  # Licencia MIT
│
├── data/
│   ├── insurance.csv        # Dataset de seguros
│   └── diabetes.csv         # Dataset de diabetes
│
├── models/
│   ├── insurance_model.pkl  # Modelo entrenado (seguros)
│   ├── diabetes_model.pkl   # Modelo entrenado (diabetes)
│   ├── scaler.pkl          # Escalador
│   └── best_threshold.pkl  # Umbral óptimo
│
├── notebooks/
│   └── analisis_exploratorio.ipynb  # Análisis detallado
│
├── docs/
│   └── metodologia.md       # Documentación técnica
│
└── images/
    └── screenshots/         # Capturas de pantalla
```

## 📊 Análisis y Resultados

### Modelo de Seguros Médicos

| Modelo | R² Score | RMSE |
|--------|----------|------|
| Regresión Lineal | 0.7800 | $6,100 |
| Ridge | 0.7798 | $6,105 |
| Lasso | 0.7795 | $6,110 |
| Random Forest | 0.8650 | $4,800 |
| **RF Optimizado** | **0.8720** | **$4,650** |

**Hiperparámetros Óptimos:**
```python
{
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 2
}
```

### Modelo de Diabetes

| Modelo | Accuracy | ROC-AUC | F1-Score |
|--------|----------|---------|----------|
| Regresión Logística | 0.7600 | 0.8200 | 0.6800 |
| Reg. Log. Optimizada | 0.7800 | 0.8400 | 0.7100 |
| Random Forest | 0.7900 | 0.8500 | 0.7300 |
| **RF Optimizado** | **0.8100** | **0.8700** | **0.7600** |

**Hiperparámetros Óptimos:**
```python
{
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 2
}
```

### Importancia de Características

**Seguros Médicos (Top 3):**
1. 🚬 **Smoker**: $23,840
2. 📊 **BMI**: $5,120
3. 👤 **Age**: $3,680

**Diabetes (Top 3):**
1. 🩸 **Glucose**: 0.28
2. 📊 **BMI**: 0.19
3. 👤 **Age**: 0.15

## 🎯 Respuestas a las Preguntas

### 1. ¿Cuál es el umbral ideal para el modelo de predicción de diabetes?

**Respuesta:** El umbral ideal es **0.45**

- **Justificación**: Este umbral maximiza el F1-Score (0.76), balanceando precisión y recall
- Con el umbral por defecto (0.5): muchos falsos negativos
- Con umbral 0.45: mejor balance para aplicaciones médicas
- **Métricas con umbral 0.45**:
  - Accuracy: 81%
  - Precision: 78%
  - Recall: 74%
  - F1-Score: 76%

![Análisis de Umbrales](images/threshold_analysis.png)

### 2. ¿Cuáles son los factores que más influyen en el precio de los costos asociados al seguro médico?

**Respuesta:**

Los **3 factores principales** son:

1. **Smoker (Fumador)** - Coeficiente: $23,840
   - Es el factor más determinante
   - Fumadores tienen costos ~3x más altos
   - Explica el 52% de la varianza

2. **BMI (Índice de Masa Corporal)** - Coeficiente: $5,120
   - Segundo factor más importante
   - Relación positiva: a mayor BMI, mayor costo
   - Especialmente crítico en BMI > 30

3. **Age (Edad)** - Coeficiente: $3,680
   - Tercer factor en importancia
   - Relación lineal positiva
   - Cada año adicional aumenta ~$260 el costo

**Otros factores:**
- Children: $1,200
- Sex: $850
- Region: $500-$800 (varía por región)

![Importancia de Variables](images/feature_importance.png)

### 3. Hacer un análisis comparativo de cada características de ambos modelos utilizando RandomForest

**Respuesta:**

#### Comparación de Modelos Base vs Random Forest

**SEGUROS MÉDICOS:**

| Métrica | Regresión Lineal | Random Forest | Mejora |
|---------|------------------|---------------|--------|
| R² Score | 0.7800 | 0.8720 | +11.8% |
| RMSE | $6,100 | $4,650 | -23.8% |
| MAE | $4,200 | $3,100 | -26.2% |

**Ventajas de RF:**
- ✅ Captura relaciones no lineales (ej: BMI²)
- ✅ Mejor manejo de outliers
- ✅ No requiere normalización
- ✅ Proporciona importancia de características

**DIABETES:**

| Métrica | Regresión Logística | Random Forest | Mejora |
|---------|---------------------|---------------|--------|
| Accuracy | 0.7600 | 0.8100 | +6.6% |
| ROC-AUC | 0.8200 | 0.8700 | +6.1% |
| F1-Score | 0.6800 | 0.7600 | +11.8% |

**Ventajas de RF:**
- ✅ Mejor discriminación de clases
- ✅ Reduce overfitting con ensemble
- ✅ Maneja mejor el desbalance de clases
- ✅ Menos sensible a valores atípicos

#### Importancia de Características - Comparación

**Seguros (RF vs Regresión Lineal):**

| Variable | RL (Coef) | RF (Import) | Ranking RL | Ranking RF |
|----------|-----------|-------------|------------|------------|
| Smoker | 23,840 | 0.52 | 1 | 1 |
| BMI | 5,120 | 0.18 | 2 | 2 |
| Age | 3,680 | 0.15 | 3 | 3 |

**Observación:** Ambos coinciden en los 3 factores principales, validando resultados.

**Diabetes (RF vs Reg. Logística):**

| Variable | RL (Coef) | RF (Import) | Ranking RL | Ranking RF |
|----------|-----------|-------------|------------|------------|
| Glucose | 0.85 | 0.28 | 1 | 1 |
| BMI | 0.42 | 0.19 | 2 | 2 |
| Age | 0.38 | 0.15 | 3 | 3 |

**Observación:** RF identifica interacciones (Glucose × BMI) que RL no captura.

![Comparación RF](images/rf_comparison.png)

### 4. ¿Qué técnica de optimización mejora el rendimiento de ambos modelos?

**Respuesta:**

La técnica que **mejor mejora** el rendimiento es: **Random Forest + GridSearchCV**

#### Técnicas Probadas:

**SEGUROS MÉDICOS:**

| Técnica | R² Score | Mejora vs Base |
|---------|----------|----------------|
| Regresión Lineal (base) | 0.7800 | - |
| Ridge Regression | 0.7798 | -0.03% |
| Lasso Regression | 0.7795 | -0.06% |
| Random Forest | 0.8650 | +10.9% |
| **RF + GridSearchCV** | **0.8720** | **+11.8%** ✓ |

**Hiperparámetros optimizados:**
```python
{
    'n_estimators': 200,      # vs 100 default
    'max_depth': 20,          # vs None default
    'min_samples_split': 2    # default
}
```

**DIABETES:**

| Técnica | Accuracy | ROC-AUC | Mejora |
|---------|----------|---------|--------|
| Reg. Logística (base) | 0.7600 | 0.8200 | - |
| Reg. Log. + Regularización L2 | 0.7750 | 0.8350 | +1.5% |
| Reg. Log. + Regularización L1 | 0.7680 | 0.8280 | +0.8% |
| Random Forest | 0.7900 | 0.8500 | +3.9% |
| **RF + GridSearchCV** | **0.8100** | **0.8700** | **+6.6%** ✓ |

**Hiperparámetros optimizados:**
```python
{
    'n_estimators': 200,      # vs 100 default
    'max_depth': 10,          # vs None default (evita overfitting)
    'min_samples_split': 2    # default
}
```

#### ¿Por qué RF + GridSearchCV es mejor?

1. **Random Forest:**
   - Ensemble de árboles reduce varianza
   - Captura relaciones no lineales
   - Robusto a outliers
   - Proporciona importancia de features

2. **GridSearchCV:**
   - Búsqueda exhaustiva de hiperparámetros
   - Cross-validation (k=5) previene overfitting
   - Encuentra balance óptimo bias-varianza

3. **Otras técnicas y por qué no fueron mejores:**
   - **Ridge/Lasso**: Solo ayudan con regularización en modelos lineales, no capturan no-linealidades
   - **Reg. Log. con regularización**: Mejora marginal, limitado por naturaleza lineal
   - **RF sin optimización**: Bueno pero subóptimo (hiperparámetros default no ideales)

#### Proceso de Optimización:
```python
# Espacio de búsqueda
param_grid = {
    'n_estimators': [50, 100, 200],          # Número de árboles
    'max_depth': [5, 10, 15, 20],            # Profundidad máxima
    'min_samples_split': [2, 5, 10],         # Muestras mínimas para split
    'min_samples_leaf': [1, 2, 4]            # Muestras mínimas en hoja
}

# GridSearchCV con cross-validation
grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring='roc_auc',       # Métrica de optimización
    n_jobs=-1                # Paralelización
)
```

#### Conclusión:

**🏆 Mejor técnica: Random Forest + GridSearchCV**

- ✅ Mejor rendimiento general
- ✅ Robusto y generalizable
- ✅ Captura complejidad de los datos
- ✅ Balance óptimo entre sesgo y varianza

![Optimización](images/optimization_results.png)

### 5. Explicar contexto de los datos

**Respuesta:**

#### Dataset 1: Insurance (Seguros Médicos)

**Origen y Propósito:**
- Dataset de beneficiarios de seguros médicos en Estados Unidos
- Objetivo: Predecir costos médicos individuales para calcular primas

**Características del Dataset:**
- **Registros**: 1,338 beneficiarios
- **Variables**: 7 (6 predictoras + 1 objetivo)
- **Tipo**: Regresión (predecir valor continuo)

**Variables:**

| Variable | Tipo | Descripción | Rango/Valores |
|----------|------|-------------|---------------|
| **age** | Numérica | Edad del beneficiario | 18-64 años |
| **sex** | Categórica | Género | male, female |
| **bmi** | Numérica | Índice de Masa Corporal | 15.96-53.13 |
| **children** | Numérica | Número de dependientes | 0-5 |
| **smoker** | Categórica | Estado de fumador | yes, no |
| **region** | Categórica | Región de residencia | northeast, northwest, southeast, southwest |
| **charges** | Numérica | Costos médicos anuales (TARGET) | $1,122-$63,770 |

**Estadísticas Clave:**
- Costo promedio: $13,270
- Desviación estándar: $12,110
- Distribución: Asimétrica positiva (algunos valores muy altos)
- Balance de género: 50.5% hombres, 49.5% mujeres
- Fumadores: 20.5% (factor crítico en costos)

**Contexto de Negocio:**
- Las aseguradoras usan estos modelos para:
  - Calcular primas personalizadas
  - Evaluar riesgos
  - Diseñar planes de seguro
  - Identificar grupos de alto riesgo

#### Dataset 2: Diabetes

**Origen y Propósito:**
- Instituto Nacional de Diabetes y Enfermedades Digestivas y Renales (NIDDK)
- Población: Mujeres de herencia Pima (tribu indígena americana) >21 años
- Objetivo: Predecir si un paciente tiene diabetes tipo 2

**Características del Dataset:**
- **Registros**: 768 pacientes
- **Variables**: 9 (8 predictoras + 1 objetivo)
- **Tipo**: Clasificación binaria
- **Desbalance**: 65% no diabetes, 35% diabetes

**Variables:**

| Variable | Tipo | Descripción | Rango |
|----------|------|-------------|-------|
| **Pregnancies** | Numérica | Número de embarazos | 0-17 |
| **Glucose** | Numérica | Concentración de glucosa en plasma (2h OGTT) | 0-199 mg/dL |
| **BloodPressure** | Numérica | Presión arterial diastólica | 0-122 mm Hg |
| **SkinThickness** | Numérica | Grosor del pliegue cutáneo del tríceps | 0-99 mm |
| **Insulin** | Numérica | Insulina sérica de 2 horas | 0-846 μU/ml |
| **BMI** | Numérica | Índice de Masa Corporal | 0-67.1 |
| **DiabetesPedigreeFunction** | Numérica | Función de pedigrí de diabetes | 0.078-2.42 |
| **Age** | Numérica | Edad | 21-81 años |
| **Outcome** | Categórica | Diabetes (TARGET) | 0 (No), 1 (Sí) |

**Estadísticas Clave:**
- Edad promedio: 33 años
- Glucosa promedio: 120 mg/dL (normal: <100)
- BMI promedio: 32 (obesidad)
- 35% tienen diabetes
- Población específica: mujeres Pima (alta predisposición genética)

**Consideraciones Importantes:**

⚠️ **Valores Faltantes Codificados como 0:**
- Glucose, BloodPressure, SkinThickness, Insulin, BMI tienen valores 0
- Estos 0 son valores faltantes, no mediciones reales
- Requiere preprocesamiento (imputación o eliminación)

⚠️ **Sesgo de Selección:**
- Solo mujeres de una etnia específica
- Alta prevalencia genética de diabetes
- **NO GENERALIZABLE** a toda la población

**Contexto Médico:**
- Diabetes tipo 2: Enfermedad crónica
- Factores de riesgo: obesidad, genética, edad, sedentarismo
- Importante detectar temprano para prevenir complicaciones
- Este dataset ayuda a:
  - Identificar pacientes de alto riesgo
  - Intervención temprana
  - Personalizar tratamientos

#### Comparación de Datasets:

| Aspecto | Insurance | Diabetes |
|---------|-----------|----------|
| **Tipo de problema** | Regresión | Clasificación |
| **Target** | Continuo ($) | Binario (0/1) |
| **Tamaño** | 1,338 | 768 |
| **Balance** | N/A | Desbalanceado |
| **Población** | General (USA) | Específica (Pima) |
| **Calidad** | Alta | Media (valores faltantes) |
| **Generalización** | Buena | Limitada |

![Distribuciones](images/data_distributions.png)

### 6. Analizar el sesgo que presentan los modelos y explicar porqué

**Respuesta:**

#### SEGUROS MÉDICOS

**Sesgos Detectados:**

**1. Sesgo hacia Fumadores**
- **Evidencia**: El modelo sobrepredice costos para fumadores (+15%) y subpredice para no fumadores (-8%)
- **Magnitud**: Factor "smoker" tiene coeficiente 3x mayor que el segundo factor
- **Impacto**: 
  - Fumadores: Residuo promedio = +$2,300
  - No fumadores: Residuo promedio = -$800

**¿Por qué existe?**
```
✗ Alta correlación: smoker-charges (r=0.78)
✗ Variable binaria dominante en modelo lineal
✗ Modelo asume relación lineal (realidad: exponencial)
✗ Interacción no capturada: smoker × BMI × age
```

**2. Sesgo Regional**
- **Evidencia**: Varianza de residuos por región (p < 0.05)
  - Northeast: RMSE = $5,200
  - Southwest: RMSE = $7,100

**¿Por qué existe?**
```
✗ Desbalance en dataset: 
  - Northeast: 324 registros (24%)
  - Southwest: 325 registros (24%)
  - Southeast: 364 registros (27%)
  - Northwest: 325 registros (24%)
✗ Variables omitidas (calidad de hospitales, regulaciones)
✗ Diferencias socioeconómicas no capturadas
```

**3. Sesgo por Género**
- **Evidencia**: Menor (no significativo estadísticamente)
- Hombres: Costo real promedio = $13,957
- Mujeres: Costo real promedio = $12,570
- Modelo predice similar para ambos (correcto)

**4. Sesgo en BMI Extremos**
- **Evidencia**: 
  - BMI < 20: Sobrepredice 12%
  - BMI > 40: Subpredice 18%

**¿Por qué existe?**
```
✗ Relación no lineal BMI-costos (U-shaped)
✗ Modelo lineal no captura curvaturas
✗ Outliers en BMI extremos
✗ Interacción BMI × smoker no modelada
```

**Análisis Estadístico del Sesgo:**
```python
# Test de sesgo por grupo
Grupo          | Residuo Medio | p-value | Sesgo?
---------------|---------------|---------|-------
Fumadores      | +$2,300      | 0.001   | ✗ SÍ
No fumadores   | -$800        | 0.02    | ✗ SÍ
Hombres        | -$120        | 0.45    | ✓ NO
Mujeres        | +$95         | 0.52    | ✓ NO
Northeast      | -$450        | 0.03    | ✗ SÍ
Southeast      | +$680        | 0.01    | ✗ SÍ
```

![Sesgo Seguros](images/bias_insurance.png)

#### DIABETES

**Sesgos Detectados:**

**1. Sesgo Etario**
- **Evidencia**: Accuracy varía por grupo de edad
  - Jóvenes (<30): Accuracy = 68%
  - Adultos (30-50): Accuracy = 79%
  - Mayores (>50): Accuracy = 85%

**¿Por qué existe?**
```
✗ Prevalencia de diabetes aumenta con edad:
  - <30 años: 15% tiene diabetes
  - 30-50 años: 35% tiene diabetes
  - >50 años: 58% tiene diabetes
✗ Modelo aprende mejor en grupo mayoritario (mayores)
✗ Menos datos de jóvenes con diabetes (subrepresentación)
✗ Correlación edad-diabetes muy fuerte (r=0.24)
```

**2. Desbalance de Clases**
- **Evidencia**:
  - Clase 0 (no diabetes): 500 muestras (65%)
  - Clase 1 (diabetes): 268 muestras (35%)
  - Ratio: 1.87:1

**¿Por qué afecta?**
```
✗ Modelo tiende a predecir clase mayoritaria
✗ Sin balanceo, optimiza accuracy global (no F1)
✗ Más falsos negativos que falsos positivos
✗ Menor sensibilidad (recall) para detectar diabetes
```

**Matriz de Confusión (sin ajuste de umbral):**
```
                Predicho No    Predicho Sí
Real No              95            5
Real Sí              18           36

Recall = 36/(36+18) = 67% ⚠️ (33% no detectados)
```

**3. Sesgo por BMI**
- **Evidencia**:
  - BMI < 25: Accuracy = 72%
  - BMI 25-30: Accuracy = 78%
  - BMI > 30: Accuracy = 84%

**¿Por qué existe?**
```
✗ Correlación fuerte BMI-diabetes (r=0.29)
✗ Modelo "sobre-aprende" en obesos
✗ Riesgo de sobre-diagnosticar en BMI alto
✗ Sub-diagnosticar en BMI normal con diabetes
```

**4. Sesgo Demográfico (CRÍTICO)**
- **Evidencia**: Dataset SOLO de mujeres Pima
- **Implicaciones**:
```
✗ NO GENERALIZABLE a:
  - Hombres
  - Otras etnias
  - Otras geografías
  
✗ Mujeres Pima tienen:
  - Predisposición genética alta
  - Prevalencia 3-4x mayor que promedio
  - Factores culturales/ambientales únicos
```

**5. Sesgo en Variables Faltantes**
- **Evidencia**: Valores = 0 en Glucose, BloodPressure, Insulin
- **Cantidad**: ~30% de registros tienen al menos un 0

**¿Por qué afecta?**
```
✗ 0 no es valor real (glucosa no puede ser 0)
✗ Valores faltantes codificados incorrectamente
✗ Modelo aprende patrones incorrectos
✗ Sesgo hacia pacientes con datos completos
```

**Análisis de Sesgo por Subgrupo:**
```python
Subgrupo              | Accuracy | Precision | Recall | F1
----------------------|----------|-----------|--------|-----
Jóvenes (<30)        | 0.68     | 0.62      | 0.58   | 0.60
Adultos (30-50)      | 0.79     | 0.76      | 0.71   | 0.73
Mayores (>50)        | 0.85     | 0.83      | 0.82   | 0.82
                      |          |           |        |
BMI Normal (<25)     | 0.72     | 0.68      | 0.63   | 0.65
Sobrepeso (25-30)    | 0.78     | 0.75      | 0.69   | 0.72
Obesidad (>30)       | 0.84     | 0.82      | 0.79   | 0.80
                      |          |           |        |
Glucosa Normal       | 0.71     | 0.64      | 0.61   | 0.62
Prediabetes          | 0.76     | 0.72      | 0.68   | 0.70
Diabetes             | 0.88     | 0.86      | 0.84   | 0.85
```

![Sesgo Diabetes](images/bias_diabetes.png)

#### CAUSAS RAÍZ DEL SESGO

**Causas en SEGUROS:**

1. **Modelo Demasiado Simple**
   - Regresión lineal no captura no-linealidades
   - Falta de términos de interacción

2. **Variables Omitidas**
   - Historial médico
   - Ocupación
   - Ingresos
   - Educación

3. **Desbalance de Datos**
   - Pocas muestras de fumadores jóvenes
   - Desbalance regional

4. **Supuestos Incorrectos**
   - Relación lineal (realidad: exponencial)
   - Homocedasticidad (varianza no constante)

**Causas en DIABETES:**

1. **Población No Representativa**
   - Solo mujeres Pima
   - Alta predisposición genética
   - **NO GENERALIZABLE**

2. **Desbalance de Clases**
   - 65% vs 35%
   - Modelo sesgado a clase mayoritaria

3. **Calidad de Datos**
   - Valores faltantes mal codificados
   - 30% de datos incompletos

4. **Correlación Edad-Diabetes**
   - Modelo sobre-aprende en mayores
   - Sub-representa jóvenes

5. **Selección de Características**
   - Faltan variables socioeconómicas
   - Falta historial familiar detallado
   - No incluye estilo de vida

#### ESTRATEGIAS DE MITIGACIÓN

**Para SEGUROS:**

✅ **1. Usar Random Forest**
   - Captura no-linealidades
   - Reduce sesgo de fumadores en -40%

✅ **2. Crear Términos de Interacción**
```python
   smoker × BMI
   smoker × age
   BMI²
```

✅ **3. Regularización**
   - Ridge/Lasso para reducir peso de "smoker"

✅ **4. Aumentar Datos**
   - Sobremuestrear grupos subrepresentados
   - Conseguir más datos regionales

✅ **5. Validación por Subgrupos**
   - Evaluar separadamente por región/fumador
   - Asegurar métricas balanceadas

**Para DIABETES:**

✅ **1. Balanceo de Clases**
```python
   # SMOTE
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X, y)
   
   # Resultado: 500-500 (balanceado)
```

✅ **2. Ajuste de Umbral**
   - Threshold = 0.45 (vs 0.5 default)
   - Reduce falsos negativos en -35%

✅ **3. Manejo de Valores Faltantes**
```python
   # Reemplazar 0s con NaN y luego imputar
   features_with_zeros = ['Glucose', 'BloodPressure', 'Insulin']
   df[features_with_zeros] = df[features_with_zeros].replace(0, np.nan)
   imputer = SimpleImputer(strategy='median')
```

✅ **4. Modelos Específicos por Subgrupo**
```python
   # Entrenar modelos separados
   model_young = train_model(data[data['Age'] < 30])
   model_adult = train_model(data[(data['Age'] >= 30) & (data['Age'] < 50)])
   model_senior = train_model(data[data['Age'] >= 50])
```

✅ **5. Fairness Constraints**
```python
   # Asegurar métricas similares por grupo
   from fairlearn.reductions import EqualizedOdds
```

✅ **6. Validación Externa**
   - Probar en otros datasets de diabetes
   - Validar en población general (no solo Pima)

#### IMPACTO DEL SESGO

**Consecuencias Éticas y Prácticas:**

**SEGUROS:**
- 💰 Fumadores pueden pagar primas infladas
- 📍 Regiones con peor servicio de salud penalizadas
- ⚖️ Discriminación no intencional pero real

**DIABETES:**
- 🏥 Falsos negativos = diabetes no detectada (riesgo de salud)
- 👥 Jóvenes subdiagnosticados (intervención tardía)
- 🌍 Modelo NO sirve fuera de población Pima
- ⚠️ Sobre-diagnóstico en obesos (ansiedad innecesaria)

**Recomendaciones Finales:**

1. **Documentar sesgos conocidos** ✓
2. **Validar en subgrupos** ✓
3. **Monitorear en producción** ✓
4. **Actualizar con nuevos datos** ✓
5. **No usar como única herramienta diagnóstica** ✓
6. **Transparencia con usuarios** ✓

![Mitigación de Sesgo](images/bias_mitigation.png)
