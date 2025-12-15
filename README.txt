ProyectoFinal
Descripción

Este proyecto utiliza DeepFace, Whisper, MoviePy, Transformers y FFmpeg para realizar análisis facial y transcripción de audio de videos. El objetivo es procesar medios e interpretar información desde archivos de audio y video.

Requisitos

Sistema Operativo: Windows 10/11 (64-bit)

Python: Versión 3.9 (64-bit)

Dependencias principales:

TensorFlow 2.13 (CPU)

DeepFace

MoviePy

OpenCV

Whisper (OpenAI)

Transformers

Pandas

SciPy

OpenXL

FFmpeg: Requerido para la manipulación de audio y video. Se debe descargar e instalar por separado.

Instalación del entorno
Paso 1: Crear un entorno virtual

Abre PowerShell o CMD en tu PC.

Navega al directorio donde tienes el proyecto:

cd C:\Users\ASUS\Downloads\ProyectoFinal\ProyectoFinal


Crea el entorno virtual:

python -m venv venv

Paso 2: Activar el entorno virtual

Para activar el entorno en PowerShell:

.\venv\Scripts\Activate.ps1


Deberías ver algo como (venv) al inicio de la línea de comando. Esto significa que el entorno virtual está activo.

Paso 3: Actualizar pip y herramientas de instalación

Para asegurar que tienes la última versión de pip, setuptools y wheel:

pip install --upgrade pip setuptools wheel

Instalación de dependencias

Con el entorno virtual activado, instala todas las dependencias necesarias:

pip install tensorflow-cpu==2.13.0 deepface moviepy opencv-python pandas scipy openpyxl transformers whisper tf-keras

Instalación de FFmpeg (requerido para Whisper y MoviePy)
Paso 1: Descargar FFmpeg

Ve a la página de FFmpeg Builds en GitHub
 y descarga el archivo .zip correspondiente para Windows 64-bit.

El archivo correcto es algo como:

ffmpeg-master-latest-win64-gpl-shared.zip

Paso 2: Extraer y configurar FFmpeg

Extrae el contenido del archivo .zip en una carpeta, por ejemplo:

C:\FFmpeg


Agrega la carpeta bin de FFmpeg al PATH del sistema:

Win + S → escribe "Editar variables de entorno del sistema" y ábrelo.

En Variables de entorno, selecciona Path y luego Editar.

Haz clic en Nuevo y agrega la ruta:

C:\FFmpeg\ffmpeg-master-latest-win64-gpl-shared\bin


Acepta los cambios y cierra.

Para verificar si FFmpeg está correctamente instalado, abre PowerShell y ejecuta:

ffmpeg -version

Verificación del entorno

Una vez que hayas instalado todo, verifica que las librerías necesarias estén funcionando correctamente ejecutando los siguientes comandos en PowerShell (con el entorno virtual activo):

python -c "import tensorflow as tf; print(tf.__version__)"
python -c "from deepface import DeepFace; print('DeepFace OK')"
python -c "from transformers import pipeline; print('Transformers OK')"
python -c "import whisper; print('Whisper OK')"


Si todo está bien instalado, deberías ver algo como:

2.13.0
DeepFace OK
Transformers OK
Whisper OK

Ejecutar el proyecto

Ahora que todo está instalado, puedes ejecutar el proyecto con el siguiente comando:

python main.py