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
from pathlib import Path
from playwright.sync_api import sync_playwright

# 파일 및 폴더 경로 (d:\Github\Unity\media\video\video 기준)
BASE_DIR = r"d:\Github\Unity\media\video\ko"
WORK_DIR = r"d:\Github\Unity\media\video\video"
MODEL_FILE = os.path.join(BASE_DIR, "model")
PROMPT_FILE = os.path.join(BASE_DIR, "prompt")

DOWNLOAD_TMP_DIR = os.path.join(WORK_DIR, "download_tmp")
FINAL_SAVE_DIR = os.path.join(WORK_DIR, "download")

os.makedirs(DOWNLOAD_TMP_DIR, exist_ok=True)
os.makedirs(FINAL_SAVE_DIR, exist_ok=True)

def extract_audio(video_path, audio_path):
    command = [
        "ffmpeg", "-y", "-i", video_path,
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
            # 하나의 프롬프트에 하나의 고정 모델(URL) 배정
            target_url = urls[idx % len(urls)]
            
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
                except Exception as e:
                    print("[에러] 입력창을 찾을 수 없습니다. (재시도)")
                    continue

                initial_download_btns = page.locator(download_btn_selector).count()
                initial_videos = page.locator("video").count()

                print("[Playwright] 기존 프롬프트 지우기 및 새 프롬프트 붙여넣기 시도 중...")
                pyperclip.copy(prompt_text)
                page.locator("div.tiptap").focus()
                page.keyboard.press("Control+A")
                time.sleep(0.5)
                page.keyboard.press("Backspace")
                time.sleep(0.5)
                page.keyboard.press("Control+V")
                time.sleep(1)

                print("[Playwright] 프롬프트 제출(생성 시작)...")
                try:
                    page.keyboard.press("Enter")
                    time.sleep(1)
                    page.locator("xpath=//button[descendant::svg]").nth(-1).click(timeout=1000)
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
