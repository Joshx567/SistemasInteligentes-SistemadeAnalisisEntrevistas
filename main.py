import os
import logging
from src import media_processor, vision_module, audio_module, analysis_core

def main():
    # ---------------- CONFIGURACIÓN ----------------
    VIDEO_NAMES = ["grupo_video1.mp4", "grupo_video2.mp4", "grupo_video3.mp4"]  # videos PROPIOS
    BASE_DIR = "data"
    RAW_VIDEO_DIR = os.path.join(BASE_DIR, "raw_videos\dia3")
    OUTPUT_DIR = BASE_DIR

    # ---------------- VALIDACIÓN DE INPUT ----------------
    missing_videos = []

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
        audio_path, frames_folder = media_processor.extract_media(video_path, BASE_DIR, video_name)

        # PASO 2: Análisis facial
        df_video = vision_module.analyze_faces(frames_folder)
        logging.info(f"Frames analizados: {len(df_video)}")

        # PASO 3: Audio + texto
        df_audio = audio_module.analyze_audio(audio_path)
        logging.info(f"Segmentos de audio detectados: {len(df_audio)}")

        # PASO 4: Integración
        df_integrated = analysis_core.synchronize_data(df_video, df_audio)

        # PASO 5: Congruencia
        final_report = analysis_core.calculate_congruence(df_integrated)

        # DÍA 4 – ANÁLISIS AVANZADO
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

    logging.info("=== PROCESAMIENTO FINALIZADO ===")

if __name__ == "__main__":
    main()
