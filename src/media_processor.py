import os
import cv2
from moviepy import VideoFileClip

def extract_media(video_path, output_base, video_name, frame_step=None):
    # Crear carpetas separadas por video
    frames_dir = os.path.join(output_base, "processed_frames\dia4", video_name)
    audio_dir = os.path.join(output_base, "processed_audio\dia4", video_name)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    # Extraer audio
    video_clip = VideoFileClip(video_path)
    audio_path = os.path.join(audio_dir, "audio.wav")
    video_clip.audio.write_audiofile(audio_path, logger=None)

    cap = cv2.VideoCapture(video_path)

    # Extraer frames (1 frame por segundo si no se especifica)
    if frame_step is None:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
    else:
        fps = frame_step
    
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
