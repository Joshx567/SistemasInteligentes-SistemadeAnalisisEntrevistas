import os
import pandas as pd
from deepface import DeepFace

def analyze_faces_full_vector(frames_folder):
    """
    Analiza frames y devuelve el vector completo de probabilidades de emociones.
    Necesario para el modelo LSTM.
    """
    results = []
    # Obtener lista de archivos ordenados
    files = [f for f in os.listdir(frames_folder) if f.endswith('.jpg')]
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    for file in files:
        img_path = os.path.join(frames_folder, file)
        second = int(file.split('_')[1].split('.')[0])
        
        try:
            # Pedimos todo el objeto
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

    return pd.DataFrame(results)

analyze_faces = analyze_faces_full_vector