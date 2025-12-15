import os
import pandas as pd
from deepface import DeepFace

def analyze_faces(frames_folder):
    results = []
    
    # Obtener lista de archivos ordenados numéricamente
    files = [f for f in os.listdir(frames_folder) if f.endswith('.jpg')]
    # Ordenar por el número en el nombre "frame_X.jpg"
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    for file in files:
        img_path = os.path.join(frames_folder, file)
        second = int(file.split('_')[1].split('.')[0])
        
        try:
            # enforce_detection=False evita error si la cara no es clara en un frame
            analysis = DeepFace.analyze(
                img_path=img_path, 
                actions=['emotion'], 
                enforce_detection=False,
                silent=True
            )
            
            # DeepFace retorna una lista, tomamos el primer rostro
            dominant_emotion = analysis[0]['dominant_emotion']
            
            results.append({
                'segundo': second,
                'emocion_facial': dominant_emotion
            })
            
        except Exception as e:
            # Si falla, asumimos "neutral" o "no_face" para no romper el flujo
            results.append({
                'segundo': second,
                'emocion_facial': 'no_detection'
            })

    return pd.DataFrame(results)