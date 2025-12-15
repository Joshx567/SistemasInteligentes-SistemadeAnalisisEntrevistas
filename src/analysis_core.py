import pandas as pd

def synchronize_data(df_video, df_audio):
    """
    Combina los datos frame a frame (segundo a segundo) con el segmento de audio correspondiente.
    """
    merged_data = []

    for _, row_vid in df_video.iterrows():
        current_sec = row_vid['segundo']
        face_emotion = row_vid['emocion_facial']
        
        # Buscar en qué segmento de audio cae este segundo
        # Filtramos donde: inicio <= segundo_actual <= fin
        match = df_audio[
            (df_audio['inicio'] <= current_sec) & 
            (df_audio['fin'] >= current_sec)
        ]
        
        text_content = ""
        text_emotion = "neutral"
        
        if not match.empty:
            # Tomamos el primer match
            text_content = match.iloc[0]['texto']
            text_emotion = match.iloc[0]['emocion_texto']
        else:
            text_content = "[Silencio]"
            text_emotion = "neutral"

        merged_data.append({
            'segundo': current_sec,
            'emocion_facial': face_emotion,
            'emocion_texto': text_emotion,
            'transcripcion': text_content
        })
        
    return pd.DataFrame(merged_data)

def calculate_congruence(df):
    """
    Determina si la emoción facial coincide con la del texto.
    """
    # Mapeo para normalizar salidas de DeepFace y RoBERTa
    # DeepFace: angry, disgust, fear, happy, sad, surprise, neutral
    # RoBERTa: anger, disgust, fear, joy, sadness, surprise, neutral
    
    map_text_to_video = {
        'anger': 'angry',
        'joy': 'happy',
        'sadness': 'sad',
        'disgust': 'disgust',
        'fear': 'fear',
        'surprise': 'surprise',
        'neutral': 'neutral'
    }

    congruencia_list = []

    for _, row in df.iterrows():
        vid_em = row['emocion_facial'].lower()
        txt_em_raw = row['emocion_texto'].lower()
        
        # Normalizar texto a terminología de video
        txt_em_norm = map_text_to_video.get(txt_em_raw, 'neutral')
        
        # Lógica de congruencia
        if vid_em == 'no_detection' or row['transcripcion'] == "[Silencio]":
            congruencia_list.append("No aplicable")
        elif vid_em == txt_em_norm:
            congruencia_list.append("Congruente")
        else:
            congruencia_list.append("INCONGRUENCIA")

    df['congruencia'] = congruencia_list
    return df