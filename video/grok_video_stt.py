import os
import random
import time
import re
import shutil
import whisper
import subprocess
import pyperclip
import cv2
import easyocr
import imageio_ffmpeg
from pathlib import Path
from playwright.sync_api import sync_playwright

# ffmpeg 경로 설정 (path에 없을 경우 대비)
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_exe)
if ffmpeg_dir not in os.environ["PATH"]:
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

# 파일 및 폴더 경로
BASE_DIR = r"c:\Users\Harry\videomake\videocreate\ko"
WORK_DIR = r"c:\Users\Harry\videomake\videocreate\video"
MODEL_FILE = os.path.join(BASE_DIR, "model")
PROMPT_FILE = os.path.join(BASE_DIR, "prompt")

DOWNLOAD_TMP_DIR = os.path.join(WORK_DIR, "download_tmp")
FINAL_SAVE_DIR = os.path.join(WORK_DIR, "download")

os.makedirs(DOWNLOAD_TMP_DIR, exist_ok=True)
os.makedirs(FINAL_SAVE_DIR, exist_ok=True)

def patch_whisper_ffmpeg():
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    # whisper 내부의 호출 경로 패치
    import whisper.audio
    
    original_load_audio = whisper.audio.load_audio
    
    def modified_load_audio(file: str, sr: int = 16000):
        try:
            # 원본 코드를 imageio_ffmpeg의 실행 파일 경로로 덮어씌워서 실행시킴 (whisper.audio.load_audio 소스와 일치하게 구현)
            cmd = [
                ffmpeg_exe,
                "-nostdin",
                "-threads", "0",
                "-i", file,
                "-f", "s16le",
                "-ac", "1",
                "-acodec", "pcm_s16le",
                "-ar", str(sr),
                "-"
            ]
            out = subprocess.run(cmd, capture_output=True, check=True).stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
            
        import numpy as np
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    whisper.audio.load_audio = modified_load_audio

patch_whisper_ffmpeg()

def extract_audio(video_path, audio_path):
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def analyze_audio_with_whisper(video_path, target_word, model):
    audio_tmp = video_path.replace(".mp4", "_temp.wav")
    try:
        extract_audio(video_path, audio_tmp)
        result = model.transcribe(
            audio_tmp, 
            language='ko', 
            initial_prompt=f"{target_word}, 동물이름", 
            verbose=False
        )
        transcribed_text = result['text'].strip()
        print(f"-> STT 결과: {transcribed_text}")
        
        # '버섯' 같은 타겟 단어가 포함되어 있는지 검증
        match = target_word in transcribed_text
        return match, transcribed_text
    finally:
        if os.path.exists(audio_tmp):
            try:
                os.remove(audio_tmp)
            except:
                pass


def check_video_for_text(video_path, reader):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(max(fps, 1)) # 1초당 1프레임 검사해서 속도 향상
    
    text_detected = False
    for i in range(0, frame_count, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
            
        results = reader.readtext(frame)
        for bbox, text, prob in results:
            if prob > 0.5 and len(text.strip()) > 0:
                print(f"      [OCR 감지됨] 화면에서 텍스트 발견: '{text}' (확률: {prob:.2f}, 프레임: {i})")
                text_detected = True
                break
                
        if text_detected:
            break
            
    cap.release()
    return text_detected


def main():
    # 1. 대상 URL, 프롬프트 준비
    with open(MODEL_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
        
    print("Loading Whisper 'small' model...")
    stt_model = whisper.load_model("small")

    print("Loading EasyOCR model for text detection...")
    ocr_reader = easyocr.Reader(['ko', 'en'])

    # 브라우저 자동화 (Playwright) 시작
    with sync_playwright() as p:
        user_data_dir = os.path.join(WORK_DIR, "grok_browser_profile")
        browser = p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=False,
            args=["--start-maximized", "--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"],
            no_viewport=True,
            accept_downloads=True,
            permissions=['clipboard-read', 'clipboard-write']
        )
        page = browser.pages[0] if browser.pages else browser.new_page()
        page.bring_to_front()
        
        # 다운로드 버튼 검사용 셀렉터
        download_btn_selector = "button:has(svg[class*='download' i]), button:has(svg[class*='Download' i]), button[aria-label*='ownload' i], button[aria-label*='다운로드']"

        for idx, prompt_text in enumerate(prompts):
            # 하나의 프롬프트(단어)마다 랜덤하게 하나의 URL을 선택합니다.
            import random
            target_url = random.choice(urls)
            
            match = re.search(r"'([^']+)'", prompt_text)
            target_word = match.group(1) if match else "버섯"
            
            print("\n" + "="*50)
            print(f"[{idx+1}/{len(prompts)}] 타겟 단어 : {target_word}")
            print(f"사용 URL  : {target_url}")
            print(f"사용 프롬프트: {prompt_text}")
            print("="*50)
            
            target_folder = os.path.join(FINAL_SAVE_DIR, target_word)
            successful_count = 0
            if os.path.exists(target_folder):
                existing_files = [f for f in os.listdir(target_folder) if f.endswith('.mp4')]
                successful_count = len(existing_files)
                if successful_count > 0:
                    print(f"[!] 이미 기존에 만들어진 영상이 {successful_count}개 존재합니다.")
                    for f in existing_files:
                        print(f"    - {f}")
            
            if successful_count >= 5:
                print(f"⏭️ [{target_word}] 이미 5개의 영상이 모두 확보되어 있어 생성을 건너뜁니다.")
                continue
            
            while successful_count < 5:
                print(f"\n▶ [{target_word}] {successful_count+1}번째 영상 생성을 시도합니다...")
                
                print(f"[Playwright] Grok 접속 중... ({target_url})")
                try:
                    page.goto(target_url, timeout=60000)
                except Exception as e:
                    print("[경고] 페이지가 완벽히 로딩되지 않았을 수도 있습니다:", e)

                try:
                    page.wait_for_selector("div.tiptap", timeout=30000)
                    
                    # Grok이 이전 세션의 프롬프트 기록(히스토리)을 자동으로 불러오는 시간이 필요합니다.
                    # 불러오기 전에 우리가 먼저 입력해버리면, 이후에 Grok이 예전 프롬프트로 덮어씌워 버립니다.
                    print("[Playwright] Grok이 이전 기록을 불러올 때까지 대기합니다...")
                    time.sleep(3.0) 
                except Exception as e:
                    print("[에러] 입력창을 찾을 수 없습니다. (재시도)")
                    continue

                initial_download_btns = page.locator(download_btn_selector).count()
                initial_videos = page.locator("video").count()

                clean_prompt = "".join(prompt_text.splitlines()).strip()
                
                print("[Playwright] 기존 텍스트 완벽 지우기 시도 중...")
                editor = page.locator("div.tiptap")
                editor.click()
                time.sleep(0.5)
                
                # 히스토리가 다 불러와졌을테니 이제 다 지웁니다.
                page.keyboard.press("Control+A")
                time.sleep(0.3)
                page.keyboard.press("Backspace")
                time.sleep(0.5)
                
                try:
                    editor.clear(timeout=1000)
                except Exception:
                    pass
                time.sleep(0.5)

                print("[Playwright] 새 프롬프트 입력...")
                pyperclip.copy(clean_prompt)
                editor.focus()
                page.keyboard.press("Control+V")
                time.sleep(1.0) # 다 들어가고 나서 충분히 여유를 줌

                print("[Playwright] 프롬프트 제출(생성 시작)...")
                # 제출 버튼: 보통 에디터 근처의 svg를 포함한 버튼임
                try:
                    # Enter 키로 제출 시도
                    page.keyboard.press("Enter")
                    time.sleep(0.5)
                    # 별도의 전송 버튼 클릭 시도 (다양한 셀렉터)
                    submit_selectors = [
                        "button[aria-label='Send message']",
                        "button:has(svg.lucide-arrow-up)",
                        "button:has(svg.lucide-rocket)",
                        "div.tiptap + div button", # 에디터 바로 뒤의 버튼
                        "button:has-text('Generate')",
                        "button:has-text('전송')"
                    ]
                    for selector in submit_selectors:
                        btn = page.locator(selector).last
                        if btn.is_visible(timeout=500):
                            btn.click()
                            break
                except Exception:
                    pass
                
                print("영상 생성을 감시합니다. (최대 2분 대기)")
                download_path = None
                new_btn_appeared = False
                generating_started = False
                rate_limited = False
                
                try:
                    for _ in range(40): # 3초 x 40번 = 120초 대기
                        time.sleep(3)
                        
                        # Rate Limit 에러 문구가 화면에 떴는지 확인
                        page_text = page.content()
                        if "Rate limit reached" in page_text or "rate limit reached" in page_text.lower():
                            print("\n[!] 🚫 Rate Limit (생성 제한) 에 도달했습니다.")
                            rate_limited = True
                            break
                        
                        current_btns = page.locator(download_btn_selector).count()
                        current_videos = page.locator("video").count()
                        
                        if current_videos == 0:
                            generating_started = True
                            
                        if generating_started and current_videos > 0:
                            new_btn_appeared = True
                            time.sleep(5) # 활성화 대기
                            break
                except Exception as wait_err:
                    print(f"\n[!] 대기 중 에러 발생 (브라우저가 닫혔을 수 있습니다): {wait_err}")
                    time.sleep(2)
                    continue

                if rate_limited:
                    print("✅ 10분간 휴식 후 다시 시도합니다...")
                    for min_left in range(10, 0, -1):
                        print(f"   남은 대기시간: {min_left}분...")
                        time.sleep(60)
                    continue # while 루프의 처음으로 돌아가서 동일 단어 다시 시도
                if new_btn_appeared:
                    print("[Playwright] 영상 완성 감지! 자동 다운로드를 시도합니다.")
                    current_btns = page.locator(download_btn_selector).count()
                    if current_btns > 0:
                        new_download_btn = page.locator(download_btn_selector).last
                        try:
                            with page.expect_download(timeout=120000) as download_info:
                                new_download_btn.click(force=True)
                            download = download_info.value
                            download_path = os.path.join(DOWNLOAD_TMP_DIR, download.suggested_filename)
                            download.save_as(download_path)
                            print(f"[전송 완료] 임시 저장: {download_path}")
                        except Exception as e:
                            print(f"[!] 자동 다운로드 실패: {e}")
                else:
                    print("[!] 시간 초과 또는 오류: 생성 완료를 감지하지 못했습니다.")
                
                # 검증 파트
                if download_path and os.path.exists(download_path):
                    print(f"[{target_word}] Whisper STT 및 OCR 텍스트 분석 시작...")
                    match, result_text = analyze_audio_with_whisper(download_path, target_word, stt_model)
                    
                    if match:
                        has_text = check_video_for_text(download_path, ocr_reader)
                        if not has_text:
                            target_folder = os.path.join(FINAL_SAVE_DIR, target_word)
                            os.makedirs(target_folder, exist_ok=True)
                            final_filename = f"{target_word}_{int(time.time())}.mp4"
                            final_filepath = os.path.join(target_folder, final_filename)
                            
                            shutil.move(download_path, final_filepath)
                            successful_count += 1
                            print(f"✅ [최종 성공] '{target_word}' (보유 영상 수: {successful_count}/5)")
                        else:
                            os.remove(download_path)
                            print(f"❌ [최종 실패] STT는 일치했으나 영상 내 자막(텍스트)이 발견됨. 삭제 후 재시도.")
                    else:
                        os.remove(download_path)
                        print(f"❌ [최종 실패] STT 불일치. 삭제 후 재시도.")
                else:
                    print("❌ [최종 실패] 분석할 동영상 파일을 확보하지 못했습니다. 재시도합니다.")
            
            print(f"🎉 [{target_word}] 5개의 조건을 만족하는 영상을 모두 확보했습니다.")

        browser.close()

if __name__ == "__main__":
    main()
