import os
import pandas as pd
from deepface import DeepFace
import cv2
import numpy as np

def frames_are_similar(img1, img2, threshold=0.95):
    """
    Compara dos frames y determina si son similares.
    
    Args:
        img1: Primer frame (numpy array)
        img2: Segundo frame (numpy array)
        threshold: Umbral de similitud (0-1)
    
    Returns:
        bool: True si los frames son similares
    """
    # Redimensionar para comparación rápida (más eficiente)
    small1 = cv2.resize(img1, (32, 32))
    small2 = cv2.resize(img2, (32, 32))
    
    # Calcular diferencia normalizada
    diff = cv2.absdiff(small1, small2)
    similarity = 1 - (np.sum(diff) / (32 * 32 * 3 * 255))
    
    return similarity >= threshold

def analyze_faces_full_vector(frames_folder, similarity_threshold=0.95):
    """
    Analiza frames y devuelve el vector completo de probabilidades de emociones.
    Necesario para el modelo LSTM.
    """
    results = []
    cache_hits = 0
    total_frames = 0
    
    # Obtener lista de archivos ordenados
    files = [f for f in os.listdir(frames_folder) if f.endswith('.jpg')]
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    previous_frame = None
    
    for file in files:
        img_path = os.path.join(frames_folder, file)
        second = int(file.split('_')[1].split('.')[0])
        total_frames += 1
        
        # Leer frame actual
        current_frame = cv2.imread(img_path)
        
        # Verificar si el frame es None (error de lectura)
        if current_frame is None:
            row = {
                'segundo': second, 
                'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 
                'sad': 0, 'surprise': 0, 'neutral': 100,
                'emocion_facial': 'no_detection'
            }
            results.append(row)
            continue
        
        # Comparar con frame anterior si existe
        if previous_frame is not None and frames_are_similar(previous_frame, current_frame, similarity_threshold):
            # Reutilizar último resultado del caché
            cached_result = results[-1].copy()
            cached_result['segundo'] = second
            results.append(cached_result)
            cache_hits += 1
            continue
        
        # Frame diferente o es el primero: hacer análisis completo
        try:
            analysis = DeepFace.analyze(
                img_path=img_path, 
                actions=['emotion'], 
                enforce_detection=False, 
                silent=True
            )
            
            # DeepFace devuelve una lista
            result_dict = analysis[0]
            emotions = result_dict['emotion'] # Diccionario {'angry': 0.1, ...}
            dominant = result_dict['dominant_emotion']
            
            # Guardamos fila con desglose numérico
            row = {'segundo': second}
            row.update(emotions) # Agrega columnas: angry, disgust, fear, happy, sad, surprise, neutral
            row['emocion_facial'] = dominant # Mantenemos la string para compatibilidad
            
            results.append(row)
            
        except Exception as e:
            # Si falla, vector neutro por defecto
            row = {
                'segundo': second, 
                'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 
                'sad': 0, 'surprise': 0, 'neutral': 100,
                'emocion_facial': 'no_detection'
            }
            results.append(row)
        
        # Actualizar frame anterior para la próxima iteración
        previous_frame = current_frame

    return pd.DataFrame(results)

# Mantener la compatibilidad con el nombre anterior
analyze_faces = analyze_faces_full_vector