import os
from src import media_processor, vision_module, lstm_model

def main():
    print("=== ENTRENAMIENTO DEL MODELO LSTM ===")
    base_dir = "data"
    # Usaremos los mismos videos del dia 3 para entrenar (Transfer Learning rápido)
    raw_video_dir = os.path.join(base_dir, "raw_videos", "dia3") 
    
    if not os.path.exists(raw_video_dir):
        print(f"Error: No existe la carpeta {raw_video_dir}")
        return

    training_dfs = []
    
    files = [f for f in os.listdir(raw_video_dir) if f.endswith('.mp4')]
    if not files:
        print("No hay videos .mp4 para entrenar.")
        return

    for video_name in files:
        print(f"--> Procesando video para entrenamiento: {video_name}")
        video_path = os.path.join(raw_video_dir, video_name)
        
        # 1. Extraer frames (usa la misma carpeta temporal)
        _, frames_folder = media_processor.extract_media(video_path, base_dir, video_name)
        
        # 2. Obtener vectores numéricos
        df = vision_module.analyze_faces_full_vector(frames_folder)
        training_dfs.append(df)
        
    # 3. Entrenar
    print("--> Iniciando entrenamiento de red neuronal...")
    lstm_model.train_and_save(training_dfs)
    print("=== ENTRENAMIENTO COMPLETADO ===")

if __name__ == "__main__":
    main()