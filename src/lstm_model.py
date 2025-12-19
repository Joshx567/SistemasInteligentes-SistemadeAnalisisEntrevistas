import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
import os
import pandas as pd

# Las 7 emociones que maneja DeepFace
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def create_lstm_model(input_shape):
    """Crea una arquitectura LSTM simple"""
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(32, activation='relu')) 
    model.add(Dense(len(EMOTIONS), activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def prepare_sequences(df, window_size=3):
    """Convierte el DataFrame en ventanas deslizantes para la LSTM."""
    # Asegurar que existan las columnas
    for col in EMOTIONS:
        if col not in df.columns:
            df[col] = 0.0

    data = df[EMOTIONS].values 
    # Normalizar de 0-100 a 0-1 si es necesario (DeepFace suele dar 0-100)
    if data.max() > 1.0:
        data = data / 100.0
    
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size]) # Input: frames anteriores
        y.append(data[i+window_size])   # Target: frame siguiente
        
    return np.array(X), np.array(y)

def train_and_save(dfs_training, model_path='data/lstm_emotion.h5'):
    window_size = 3
    X_all, y_all = [], []
    
    for df in dfs_training:
        if len(df) > window_size:
            X, y = prepare_sequences(df, window_size)
            X_all.append(X)
            y_all.append(y)
            
    if not X_all:
        print("ERROR: No hay suficientes datos para entrenar.")
        return

    X_train = np.concatenate(X_all)
    y_train = np.concatenate(y_all)
    
    print(f"Entrenando LSTM con {len(X_train)} secuencias...")
    model = create_lstm_model((window_size, 7))
    model.fit(X_train, y_train, epochs=15, batch_size=4, verbose=1)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Modelo guardado en {model_path}")

def predict_sequence(df, model_path='data/lstm_emotion.h5', window_size=3):
    """Usa el modelo guardado para suavizar las emociones"""
    if not os.path.exists(model_path):
        print("Modelo LSTM no encontrado.")
        return df['emocion_facial'].tolist() # Retorna original si falla

    model = load_model(model_path)
    
    # Preparar datos
    for col in EMOTIONS:
        if col not in df.columns:
            df[col] = 0.0
            
    data = df[EMOTIONS].values
    if data.max() > 1.0: data = data / 100.0

    sequences = []
    # Rllenar padding para mantener longitud
    for i in range(len(data)):
        if i < window_size:
            seq = np.array([data[0]] * window_size)
        else:
            seq = data[i-window_size : i]
        sequences.append(seq)
    
    preds = model.predict(np.array(sequences), verbose=0)
    indices = np.argmax(preds, axis=1)
    labels = [EMOTIONS[i] for i in indices]
    
    return labels