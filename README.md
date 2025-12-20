# Sistema de Análisis Multimodal de Entrevistas (S.A.M.E.)

![Python](https://img.shields.io/badge/Python-3.9-blue) ![DeepFace](https://img.shields.io/badge/AI-DeepFace-yellow) ![Whisper](https://img.shields.io/badge/AI-Whisper-green)

## Descripción del Proyecto
Este sistema implementa un pipeline de Inteligencia Artificial "End-to-End" diseñado para analizar entrevistas laborales o psicológicas. Utiliza un enfoque multimodal que combina:
1.  **Visión Computacional (CNN):** Detección de microexpresiones faciales frame a frame.
2.  **Procesamiento de Audio (ASR):** Transcripción de alta fidelidad con Whisper.
3.  **Procesamiento de Lenguaje Natural (Transformers):** Análisis de sentimiento sobre el texto hablado.
4.  **Análisis Temporal:** Algoritmos de suavizado (Rolling Window / LSTM) para coherencia en series de tiempo.

El sistema genera un **Reporte de Congruencia** que detecta discrepancias entre lo que el usuario *dice* (texto) y lo que su rostro *expresa* (video).

## 🛠️ Stack Tecnológico
| Componente | Tecnología Implementada | Función |
|------------|-------------------------|---------|
| **Visión** | `DeepFace` (Wrapper TensorFlow) | Extracción de emociones (FER-2013) |
| **Audio** | `OpenAI Whisper` (Base) | ASR (Speech to Text) |
| **NLP** | `Transformers` (DistilRoBERTa) | Clasificación de emociones en texto |
| **Temporal** | `LSTM de TensorFlow` | Análisis de series temporales y suavizado |
| **Gráficos** | `Matplotlib` / `Seaborn` | Visualización de incongruencias |

## 🧠 Arquitectura del Sistema

### Módulos principales
- **media_processor**  
  Extrae audio y frames (1 fps) desde los videos.

- **vision_module**  
  Detecta emociones faciales frame por frame usando DeepFace.

- **audio_module**  
  Transcribe audio con Whisper y clasifica emociones del texto usando Transformers.

- **analysis_core**  
  Sincroniza audio y video, calcula congruencia emocional, detecta cambios emocionales y genera insights.

- **lstm_model**  
  Modelo LSTM para analisis temporal de emociones.

- **main.py**  
  Orquesta el pipeline completo end-to-end.

---

## 🎥 Dataset de Validación
- 3–5 videos propios grabados por el equipo
- Duración aproximada: 30–60 segundos
- Ubicación: `data/raw_videos/dia3/`

---

## 🛠 Requisitos

- Sistema Operativo: Windows 10/11 (64-bit)
- Python 3.9 (64-bit)
- FFmpeg (configurado en PATH)

## 🚀 Instalación y Configuración

### Prerrequisitos
*   Windows 10/11 (64-bit)
*   Python 3.9+
*   **ffmpeg** (Esencial para procesamiento de audio)

### Paso 1: Configurar FFmpeg
1. Descargar [FFmpeg Builds](https://github.com/BtbN/FFmpeg-Builds/releases).
2. Extraer en `C:\ffmpeg`.
3. Agregar `C:\ffmpeg\bin` a las Variables de Entorno (PATH).
4. Verificar en terminal: `ffmpeg -version`.

### Paso 2: Instalación del Entorno

# 1. Crear entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -r requirements.txt

### Dependencias principales
tensorflow-cpu==2.13.0
deepface
moviepy
opencv-python
pandas
scipy
openpyxl
transformers
whisper
tf-keras

yaml
---

## ⚙️ Instalación
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt


## 📊 Salidas del Sistema
Para cada video se generan:

- Reporte CSV (`report_day4_<video>.csv`)
- Reporte JSON (`report_day4_<video>.json`)

Campos principales:
- segundo
- emocion_facial
- emocion_texto
- transcripcion
- congruencia
- cambio_emocional

---

## Ejecución

python main.py

# Video Demostración

https://www.youtube.com/watch?v=QdDGvK2JHYI   

Incluye explicación del sistema, ejecución y análisis de resultados.
