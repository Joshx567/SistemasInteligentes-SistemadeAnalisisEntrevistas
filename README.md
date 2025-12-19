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
| **Temporal** | `Pandas Rolling Window` | Análisis de series temporales y suavizado |
| **Gráficos** | `Matplotlib` / `Seaborn` | Visualización de incongruencias |

## 🚀 Instalación y Configuración

### Prerrequisitos
*   Windows 10/11 (64-bit)
*   Python 3.9+
*   **FFmpeg** (Esencial para procesamiento de audio)

### Paso 1: Configurar FFmpeg
1. Descargar [FFmpeg Builds](https://github.com/BtbN/FFmpeg-Builds/releases).
2. Extraer en `C:\FFmpeg`.
3. Agregar `C:\FFmpeg\bin` a las Variables de Entorno (PATH).
4. Verificar en terminal: `ffmpeg -version`.

### Paso 2: Instalación del Entorno
```bash
# 1. Crear entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -r requirements.txt