import whisper
import os
import subprocess

def extract_audio(video_path, audio_path):
    """Extract audio from video using ffmpeg."""
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def verify_videos():
    # Load a larger, more accurate model ('small' or 'medium')
    # 'small' is a good balance between speed and accuracy for CPU
    print("Loading Whisper 'small' model for higher accuracy...")
    model = whisper.load_model("small")
    
    videos = ["bird.mp4", "bird1.mp4", "cat.mp4"]
    word_map = {
        "bird": ["새", "bird"],
        "cat": ["고양이", "cat"]
    }
    results = []

    for video in videos:
        if not os.path.exists(video):
            continue
            
        audio_tmp = video.replace(".mp4", "_v4.wav")
        print(f"Deep analyzing {video}...")
        
        extract_audio(video, audio_tmp)
        
        # Transcribe with 'initial_prompt' to guide the model to use specific Korean words
        # This helps a lot with short words like '새'
        result = model.transcribe(
            audio_tmp, 
            language='ko', 
            initial_prompt="새, 고양이, 동물이름", 
            verbose=False
        )
        transcribed_text = result['text'].strip()
        
        if os.path.exists(audio_tmp):
            os.remove(audio_tmp)
            
        prefix = video.split('.')[0].replace('1', '')
        target_words = word_map.get(prefix, [prefix])
        
        # Check if target word exists in transcription
        match = any(word in transcribed_text for word in target_words)
        
        results.append({
            "file": video,
            "text": transcribed_text,
            "match": "Yes" if match else "No"
        })

    # Output detailed results
    with open("stt_results_final.txt", "w", encoding="utf-8") as f:
        f.write("\n" + "="*60 + "\n")
        f.write(f"{'File Name':<15} | {'High Precision Result':<30} | {'Match'}\n")
        f.write("-" * 60 + "\n")
        for res in results:
            f.write(f"{res['file']:<15} | {res['text']:<30} | {res['match']}\n")
        f.write("="*60 + "\n")
    
    print("Higher precision results saved to stt_results_final.txt")

if __name__ == "__main__":
    verify_videos()
