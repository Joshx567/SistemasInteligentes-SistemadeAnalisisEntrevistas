import whisper
import pandas as pd
from transformers import pipeline

#Whisper (ASR), Transformers (RoBERTa emociones) 
def analyze_audio(audio_path):
    # 1. Cargar Whisper (ASR)
    # 'base' es un buen balance entre velocidad y precisión para CPU
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=False) # fp16=False para evitar warnings en CPU
    
    # 2. Cargar Modelo de Emociones (NLP)
    # Usamos uno específico para emociones (RoBERTa)
    emotion_classifier = pipeline(
        "text-classification", 
        model="j-hartmann/emotion-english-distilroberta-base", 
        top_k=1
    )

    data = []
    
    for segment in result['segments']:
        text = segment['text']
        start = segment['start']
        end = segment['end']
        
        # Clasificar emoción del texto
        # Truncation=True por si el texto es muy largo
        emotion_prediction = emotion_classifier(text, truncation=True)
        emotion_label = emotion_prediction[0][0]['label']
        confidence = emotion_prediction[0][0]['score']

        data.append({
            'inicio': start,
            'fin': end,
            'texto': text.strip(),
            'emocion_texto': emotion_label,
            'confianza_texto': confidence
        })
        
    return pd.DataFrame(data)