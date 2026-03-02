import easyocr
import cv2
import sys

def check_video_for_text(video_path):
    reader = easyocr.Reader(['ko', 'en'])
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Analyzing {video_path}, FPS: {fps}, Total frames: {frame_count}")
    
    # Check 1 frame per second to save time
    frame_interval = int(max(fps, 1))
    
    text_found = False
    for i in range(0, frame_count, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
            
        print(f"Checking frame {i}...")
        results = reader.readtext(frame)
        
        for bbox, text, prob in results:
            if prob > 0.5 and len(text.strip()) > 0:
                print(f"TEXT DETECTED at frame {i}: '{text}' (prob: {prob:.2f})")
                text_found = True
                
        if text_found:
            break
            
    cap.release()
    print("Finished checking video." if not text_found else "Video rejected due to text.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_video_for_text(sys.argv[1])
    else:
        print("Provide video path")
