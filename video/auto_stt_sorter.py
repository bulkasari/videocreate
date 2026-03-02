import os
import time
import shutil
import whisper
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 설정 값
DOWNLOAD_DIR = r"C:\Users\bulka\Downloads"
SORT_DIR_BASE = r"d:\Github\Unity\media\video\video\download"
TARGET_WORD_KO = "버섯"
TARGET_WORD_EN = "mushroom"
TARGET_WORDS = ["버섯", "버선", "벚꽃", "벗", "버", "설", "섯"] # 버섯으로 발음될 수 있는 유사 발음 포함

print("Loading Whisper 'small' model...")
model = whisper.load_model("small")
print("Model loaded successfully. Monitoring downloads folder...")

def extract_audio(video_path, audio_path):
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class DownloadHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".mp4"):
            print(f"\n[+] 새 영상 파일 감지됨: {event.src_path}")
            # 다운로드가 완료될 때까지 잠시 대기
            time.sleep(2)
            self.process_video(event.src_path)
            
    def process_video(self, video_path):
        audio_tmp = video_path.replace(".mp4", "_temp.wav")
        
        try:
            print(f"[{TARGET_WORD_KO}] 오디오 추출 중...")
            extract_audio(video_path, audio_tmp)
            
            print(f"[{TARGET_WORD_KO}] Whisper STT 분석 중...")
            result = model.transcribe(
                audio_tmp, 
                language='ko', 
                initial_prompt=f"{TARGET_WORD_KO}, 동물이름", 
                verbose=False
            )
            transcribed_text = result['text'].strip()
            print(f"-> STT 결과: {transcribed_text}")
            
            # 발음 매칭 확인 (지정된 폴더로 할당)
            match = any(word in transcribed_text for word in TARGET_WORDS)
            
            if match:
                folder_path = os.path.join(SORT_DIR_BASE, TARGET_WORD_KO)
                os.makedirs(folder_path, exist_ok=True)
                new_filename = f"generated_{TARGET_WORD_EN}_{int(time.time())}.mp4"
                new_file_path = os.path.join(folder_path, new_filename)
                
                print(f"[매칭 성공!] 파일을 이동합니다: {new_file_path}")
                shutil.move(video_path, new_file_path)
            else:
                print(f"[매칭 실패] '{TARGET_WORD_KO}' 단어를 찾지 못했습니다. 파일을 삭제합니다.")
                os.remove(video_path)
                
        except Exception as e:
            print(f"오류 발생: {e}")
        finally:
            if os.path.exists(audio_tmp):
                os.remove(audio_tmp)

if __name__ == "__main__":
    os.makedirs(SORT_DIR_BASE, exist_ok=True)
    event_handler = DownloadHandler()
    observer = Observer()
    observer.schedule(event_handler, DOWNLOAD_DIR, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
