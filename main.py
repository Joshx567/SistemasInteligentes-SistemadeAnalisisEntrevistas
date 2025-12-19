import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin" ## Esto basicamente por si no encuentra Ffmpeg en las variables y ruta
from src import media_processor, vision_module, audio_module, analysis_core

def main():
    # ---------------- CONFIGURACIÓN ----------------
    VIDEO_NAMES = ["grupo_video1.mp4", "grupo_video2.mp4", "grupo_video3.mp4"]  # videos PROPIOS
    BASE_DIR = "data"
    RAW_VIDEO_DIR = os.path.join(BASE_DIR, "raw_videos", "dia3")
    OUTPUT_DIR = BASE_DIR

    # ---------------- VALIDACIÓN DE INPUT ----------------
    missing_videos = []

    if not os.path.exists(RAW_VIDEO_DIR):
         logging.error(f"El directorio {RAW_VIDEO_DIR} no existe.")
         return
    
    for video_name in VIDEO_NAMES:
        video_path = os.path.join(RAW_VIDEO_DIR, video_name)
        if not os.path.exists(video_path):
            missing_videos.append(video_name)

    if missing_videos:
        logging.error("No se encontraron los siguientes videos:")
        for v in missing_videos:
            logging.error(f" - {v}")
        logging.error("Corrige los archivos antes de ejecutar el sistema.")
        return

    # ---------------- LOGGING ----------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("=== INICIANDO SISTEMA DE ANÁLISIS DE ENTREVISTAS ===")

    # ---------------- PROCESAR CADA VIDEO ----------------
    for video_name in VIDEO_NAMES:
        video_path = os.path.join(RAW_VIDEO_DIR, video_name)

        if not os.path.exists(video_path):
            logging.warning(f"Video no encontrado: {video_name}")
            continue

        logging.info(f"Procesando video: {video_name}")

        # PASO 1: Procesamiento de medios
        # Nota: Asegúrate de que tu función extract_media acepte estos 3 argumentos
        audio_path, frames_folder = media_processor.extract_media(video_path, BASE_DIR, video_name)

        # PASO 2: Análisis facial
        df_video = vision_module.analyze_faces(frames_folder)
        logging.info(f"Frames analizados: {len(df_video)}")

        # PASO 3: Audio + texto
        df_audio = audio_module.analyze_audio(audio_path)
        logging.info(f"Segmentos de audio detectados: {len(df_audio)}")

        # PASO 4: Integración
        df_integrated = analysis_core.synchronize_data(df_video, df_audio)

        # --- LO QUE FALTABA 2: ANÁLISIS TEMPORAL (REQUISITO DÍA 4) ---
        logging.info("Aplicando Análisis Temporal (Series de Tiempo)...")
        # Intenta usar LSTM si existe, sino usa Suavizado, sino sigue con datos crudos
        if hasattr(analysis_core, 'apply_lstm_smoothing'):
            df_integrated = analysis_core.apply_lstm_smoothing(df_integrated)
            logging.info("-> Modelo LSTM aplicado.")
        elif hasattr(analysis_core, 'apply_temporal_smoothing'):
            df_integrated = analysis_core.apply_temporal_smoothing(df_integrated)
            logging.info("-> Suavizado temporal aplicado.")
        else:
            logging.warning("-> No se encontró función de suavizado. Usando datos crudos.")
        # -------------------------------------------------------------

        # PASO 5: Congruencia
        final_report = analysis_core.calculate_congruence(df_integrated)

        # DÍA 4 – ANÁLISIS AVANZADO EXTRA
        final_report = analysis_core.detect_emotional_changes(final_report)

        metrics = analysis_core.compute_congruence_metrics(final_report)
        insights = analysis_core.generate_insights(final_report)

        logging.info(f"Métricas: {metrics}")
        logging.info(f"Insights: {insights}")

        # ---------------- SALIDAS ----------------
        csv_path = os.path.join(OUTPUT_DIR, f"report_day4_{video_name}.csv")
        json_path = os.path.join(OUTPUT_DIR, f"report_day4_{video_name}.json")

        final_report.to_csv(csv_path, index=False)
        final_report.to_json(json_path, orient="records", indent=2)

        logging.info(f"Reporte generado: {csv_path}")
        logging.info(f"Reporte JSON generado: {json_path}")

        logging.info("Generando visualización gráfica...")
        path_img = generar_grafica_avanzada(final_report, video_name, OUTPUT_DIR)
        logging.info(f"Gráfico guardado en: {path_img}")

    logging.info("=== PROCESAMIENTO FINALIZADO ===")

def generar_grafica_avanzada(df, video_name, output_dir):
    """
    Genera un gráfico comparativo de emociones Video vs Texto a lo largo del tiempo.
    Marca visualmente las incongruencias.
    """
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")

    # Mapeo de emociones a valores numéricos para poder graficar
    emotion_map = {
        'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3, 
        'fear': 4, 'surprise': 5, 'disgust': 6, 'no_detection': -1
    }
    
    # Crear columnas numéricas temporales para el gráfico
    df['video_val'] = df['emocion_facial'].apply(lambda x: emotion_map.get(str(x).lower(), 0))
    df['text_val'] = df['emocion_texto'].apply(lambda x: emotion_map.get(str(x).lower(), 0))

    # Lineplot de la emoción facial
    sns.lineplot(data=df, x='segundo', y='video_val', label='Emoción Facial (Video)', 
                 color='blue', linewidth=2, marker='o', markersize=4)

    # Scatterplot de la emoción del texto (Solo puntos donde hay habla)
    text_data = df[df['text_val'] != 0]
    if not text_data.empty:
        plt.scatter(text_data['segundo'], text_data['text_val'], 
                    color='green', s=100, label='Emoción Texto', marker='s', zorder=5)

    # Marcar Incongruencias
    incongruencias = df[df['congruencia'] == 'INCONGRUENCIA']
    if not incongruencias.empty:
        plt.scatter(incongruencias['segundo'], incongruencias['video_val'], 
                    color='red', s=150, label='Incongruencia', marker='X', zorder=10)

    # Formato del eje Y
    plt.yticks(list(emotion_map.values()), list(emotion_map.keys()))
    plt.ylim(-1.5, 7)
    
    plt.title(f"Análisis Multimodal: {video_name}", fontsize=16)
    plt.xlabel("Tiempo (segundos)", fontsize=12)
    plt.ylabel("Emoción Detectada", fontsize=12)
    plt.legend(loc='upper right')
    
    # Guardar
    output_path = os.path.join(output_dir, f"grafico_{video_name}.png")
    plt.savefig(output_path)
    plt.close()
    return output_path

if __name__ == "__main__":
    main()