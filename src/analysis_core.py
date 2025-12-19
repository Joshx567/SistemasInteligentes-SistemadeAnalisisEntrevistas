import pandas as pd
import numpy as np
from src import lstm_model 

def synchronize_data(df_video, df_audio):
    """
    Combina los datos frame a frame (segundo a segundo) con el segmento de audio correspondiente.
    """
    merged_data = []

    for _, row_vid in df_video.iterrows():
        current_sec = row_vid['segundo']
        
        # Guardamos TODO el vector numérico también en el merged (si existe)
        row_data = row_vid.to_dict()
        
        # Buscar en qué segmento de audio cae este segundo
        match = df_audio[
            (df_audio['inicio'] <= current_sec) & 
            (df_audio['fin'] >= current_sec)
        ]
        
        if not match.empty:
            # Tomamos el primer match
            row_data['texto'] = match.iloc[0]['texto']
            row_data['emocion_texto'] = match.iloc[0]['emocion_texto']
        else:
            row_data['texto'] = "[Silencio]"
            row_data['emocion_texto'] = "neutral"

        merged_data.append(row_data)
        
    return pd.DataFrame(merged_data)

def apply_lstm_smoothing(df):
    """
    Usa el modelo LSTM entrenado para predecir emociones suavizadas.
    """
    print("   [LSTM] Ejecutando inferencia de series temporales...")
    
    # Llamamos a la función predict_sequence del módulo lstm_model
    try:
        new_emotions = lstm_model.predict_sequence(df)
        
        # Guardar original y nueva
        df['emocion_facial_raw'] = df['emocion_facial']
        df['emocion_facial'] = new_emotions # Sobrescribir para que el resto use la suavizada
        
    except Exception as e:
        print(f"   [ADVERTENCIA] Falló LSTM ({e}). Usando datos originales.")
    
    return df

def calculate_congruence(df):
    """
    Determina si la emoción facial coincide con la del texto.
    """
    map_text_to_video = {
        'anger': 'angry', 'joy': 'happy', 'sadness': 'sad',
        'disgust': 'disgust', 'fear': 'fear', 'surprise': 'surprise',
        'neutral': 'neutral'
    }

    congruencia_list = []

    for _, row in df.iterrows():
        # Manejo seguro por si no existen las columnas
        vid_em = str(row.get('emocion_facial', 'neutral')).lower()
        txt_em_raw = str(row.get('emocion_texto', 'neutral')).lower()
        
        # Normalizar texto a terminología de video
        txt_em_norm = map_text_to_video.get(txt_em_raw, 'neutral')
        
        # Lógica de congruencia
        if vid_em == 'no_detection' or row.get('texto') == "[Silencio]":
            congruencia_list.append("No aplicable")
        elif vid_em == txt_em_norm:
            congruencia_list.append("Congruente")
        else:
            congruencia_list.append("INCONGRUENCIA")

    df['congruencia'] = congruencia_list
    return df

def detect_emotional_changes(df):
    """Detecta cambios bruscos de emoción."""
    df['cambio_emocion'] = df['emocion_facial'].ne(df['emocion_facial'].shift())
    return df

def compute_congruence_metrics(df):
    """Calcula métricas resumen para el reporte."""
    total = len(df)
    if total == 0: return {}
    
    incongruencias = df[df['congruencia'] == 'INCONGRUENCIA'].shape[0]
    porcentaje = (incongruencias / total) * 100
    
    return {
        "total_frames": total,
        "incongruencias_detectadas": incongruencias,
        "porcentaje_incongruencia": round(porcentaje, 2)
    }

def generate_insights(df):
    """Genera texto automático para el reporte."""
    metrics = compute_congruence_metrics(df)
    if not metrics: return "Sin datos."
    
    if metrics['porcentaje_incongruencia'] > 20:
        return "ALERTA: Alta tasa de incongruencia. Posible engaño o nerviosismo."
    else:
        return "El sujeto muestra consistencia entre expresiones faciales y discurso."