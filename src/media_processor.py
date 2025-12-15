import cv2
import os
from moviepy import VideoFileClip

def extract_media(video_path, output_base):
    # Crear carpetas si no existen
    frames_dir = os.path.join(output_base, "processed_frames")
    audio_dir = os.path.join(output_base, "processed_audio")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    # 1. Extraer Audio
    video_clip = VideoFileClip(video_path)
    audio_path = os.path.join(audio_dir, "temp_audio.wav")
    # Escribir audio (logger=None para que no llene la consola de spam)
    video_clip.audio.write_audiofile(audio_path, logger=None)

    # 2. Extraer Frames (1 frame por segundo)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Guardar frame cada 1 segundo aprox (modulo FPS)
        if count % fps == 0:
            frame_name = os.path.join(frames_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1
            
        count += 1
        
    cap.release()
    video_clip.close()
    
    return audio_path, frames_dir