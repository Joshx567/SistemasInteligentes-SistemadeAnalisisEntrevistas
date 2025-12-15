import os
from src import media_processor, vision_module, audio_module, analysis_core

def main():
    # CONFIGURACIÓN
    VIDEO_NAME = "1.mp4"
    BASE_DIR = "data"
    VIDEO_PATH = os.path.join(BASE_DIR, "raw_videos", VIDEO_NAME)
    
    # Verificar si existe el video
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: No se encuentra el video en {VIDEO_PATH}")
        return

    print("=== INICIANDO SISTEMA DE ANÁLISIS DE ENTREVISTAS ===")

    # PASO 1: Procesamiento de Medios (Día 1)
    print("\n[1/5] Extrayendo frames y audio...")
    audio_path, frames_folder = media_processor.extract_media(VIDEO_PATH, BASE_DIR)
    
    # PASO 2: Análisis Visual (Día 2)
    print("\n[2/5] Analizando expresiones faciales (DeepFace)...")
    df_video = vision_module.analyze_faces(frames_folder)
    print(f"   -> Frames analizados: {len(df_video)}")

    # PASO 3: Análisis de Audio y Texto (Día 2)
    print("\n[3/5] Transcribiendo y analizando sentimiento del texto (Whisper + Transformers)...")
    df_audio = audio_module.analyze_audio(audio_path)
    print(f"   -> Segmentos de texto detectados: {len(df_audio)}")

    # PASO 4: Integración Multimodal (Día 3)
    print("\n[4/5] Sincronizando y fusionando datos...")
    df_integrated = analysis_core.synchronize_data(df_video, df_audio)

    # PASO 5: Generación de Insights (Día 4)
    print("\n[5/5] Generando reporte de congruencia...")
    final_report = analysis_core.calculate_congruence(df_integrated)

    # RESULTADOS
    output_csv = os.path.join(BASE_DIR, "reporte_final.csv")
    final_report.to_csv(output_csv, index=False)
    
    print("\n" + "="*50)
    print(f"¡ÉXITO! Reporte generado en: {output_csv}")
    print("="*50)
    print(final_report[['segundo', 'emocion_facial', 'emocion_texto', 'congruencia']].head(10))

if __name__ == "__main__":
    main()