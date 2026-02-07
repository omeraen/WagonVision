import cv2
import os
import numpy as np
import torch
import pathlib
from ultralytics import YOLO
from collections import defaultdict
import warnings
import time

VIDEO_SOURCE = './test.mp4'

SAVE_RESULT = False
OUTPUT_FILENAME = 'realtime_result.mp4'

warnings.filterwarnings("ignore", category=FutureWarning)
temp = pathlib.PosixPath
pathlib.WindowsPath = pathlib.PosixPath

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VAGON_NUMBER_DETECTION = 'vagonNumberDetection_2.pt'
VAGON_NUMBER_CLASSIFICATION = 'vagonNumberClassification_v8n.pt'

FINAL_OUTPUT_PATH = os.path.join(BASE_DIR, OUTPUT_FILENAME)

DIGIT_CONF = 0.30
ZONE_TOLERANCE = 0.22
CROP_PADDING = 70
MIN_FRAMES_TO_SURVIVE = 15
MIN_CONF_TO_FIX = 0.60
FRAMES_TO_FORGET = 45

def calculate_checksum(number_str_7):
    if len(number_str_7) != 7: return -1
    digits = [int(d) for d in number_str_7]
    weights = [2, 1, 2, 1, 2, 1, 2]
    total_sum = 0
    for d, w in zip(digits, weights):
        res = d * w
        total_sum += (res // 10) + (res % 10)
    next_ten = (total_sum + 9) // 10 * 10
    return next_ten - total_sum

def validate_wagon_number(number_str):
    if len(number_str) != 8 or not number_str.isdigit(): return False
    target = int(number_str[-1])
    calc = calculate_checksum(number_str[:7])
    return target == calc

def repair_number(detected_digits):
    detected_digits.sort(key=lambda x: x[0])
    original_str = "".join([d[1] for d in detected_digits])
    if len(original_str) != 8: return original_str, False, "Len!=8"
    if validate_wagon_number(original_str): return original_str, False, "Valid"
    prefix = original_str[:7]
    last_digit_conf = detected_digits[7][2]
    
    if last_digit_conf < 0.90:
        correct_last = calculate_checksum(prefix)
        candidate = prefix + str(correct_last)
        return candidate, True, f"Fixed Last: {original_str[-1]}->{correct_last}"
    
    min_conf_val = 1.0
    min_conf_idx = -1
    for i in range(7):
        if detected_digits[i][2] < min_conf_val:
            min_conf_val = detected_digits[i][2]
            min_conf_idx = i
            
    if min_conf_val < MIN_CONF_TO_FIX and min_conf_idx != -1:
        current_tail = int(original_str[-1])
        for d in range(10):
            temp_prefix = list(prefix)
            temp_prefix[min_conf_idx] = str(d)
            temp_str = "".join(temp_prefix)
            if calculate_checksum(temp_str) == current_tail:
                new_full = temp_str + str(current_tail)
                return new_full, True, f"Brute pos {min_conf_idx}: {original_str[min_conf_idx]}->{d}"

    return original_str, False, "No Fix"

def process_finished_track(track_id, scores_dict, lifespan):
    if lifespan < MIN_FRAMES_TO_SURVIVE: return
    if not scores_dict: return
    sorted_candidates = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
    winner, score = sorted_candidates[0]
    is_valid = validate_wagon_number(winner)
    threshold = 5.0
    if lifespan > 40: threshold = 2.0
    if lifespan > 80: threshold = 0.5
    if score < threshold: return
    status = "✅ VALID" if is_valid else "❌ INVALID"
    print(f"\n[EVENT] 🚆 Wagon ID:{track_id} | Number: {winner} | {status}")
    
    if not is_valid and len(sorted_candidates) > 1:
        runner_up, r_score = sorted_candidates[1]
        if validate_wagon_number(runner_up.replace("°", "")) and r_score > (score * 0.1):
            print(f"        ↳ Alternative: {runner_up} ({r_score:.1f}) ✅")
    print("-" * 50)

def run_realtime():
    print("📡 Loading detector model (YOLOv8)...")
    try:
        detector = YOLO(VAGON_NUMBER_DETECTION)
    except Exception as e:
        print(f"Error loading detector: {e}")
        return

    print("⏳ Loading digit detector (YOLOv8)...")
    try:
        classifier = YOLO(VAGON_NUMBER_CLASSIFICATION)
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return
    
    if isinstance(VIDEO_SOURCE, int):
        print(f"📷 Attempting to open camera...")
        cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 
        cap.set(cv2.CAP_PROP_FPS, 10.0)
    else:
        cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        print("❌ Failed to open video source!")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"✅ Camera started: {width}x{height} @ {fps} FPS")

    out = None
    if SAVE_RESULT: out = cv2.VideoWriter(FINAL_OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_center_x = width / 2
    zone_pixel_width = width * ZONE_TOLERANCE
    zone_left = int(frame_center_x - zone_pixel_width)
    zone_right = int(frame_center_x + zone_pixel_width)    
    track_scores = defaultdict(lambda: defaultdict(float))
    track_lifespan = defaultdict(int) 
    last_seen_frame = {}
    frame_count = 0
    print("🚀 Started. Press 'q' to exit.")
    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️ Stream interrupted or camera disconnected.")
            break
        frame_count += 1
        results = detector.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
        result = results[0]
        cv2.line(frame, (zone_left, 0), (zone_left, height), (200, 200, 200), 2)
        cv2.line(frame, (zone_right, 0), (zone_right, height), (200, 200, 200), 2)
        current_frame_track_ids = []
        
        if result.boxes and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                current_frame_track_ids.append(track_id)
                track_lifespan[track_id] += 1
                last_seen_frame[track_id] = frame_count
                x1_orig, y1_orig, x2_orig, y2_orig = map(int, box)
                box_center_x = (x1_orig + x2_orig) / 2
                is_stable = track_lifespan[track_id] > 5
                is_in_zone = zone_left < box_center_x < zone_right
                color = (0, 0, 255)
                
                if is_in_zone:
                    color = (0, 255, 0)
                    x1_pad = max(0, x1_orig - CROP_PADDING)
                    y1_pad = max(0, y1_orig - CROP_PADDING)
                    x2_pad = min(width, x2_orig + CROP_PADDING)
                    y2_pad = min(height, y2_orig + CROP_PADDING)
                    crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                    
                    if crop.size > 0:
                        results_v8 = classifier.predict(crop, conf=DIGIT_CONF, iou=0.45, verbose=False)
                        predictions = results_v8[0].boxes.data.cpu().numpy()
                        detected_digits = []
                        
                        for pred in predictions:
                            dx1, dy1, dx2, dy2, d_conf, d_cls = pred
                            class_id = int(d_cls)
                            
                            if hasattr(classifier, 'names'):
                                names = classifier.names
                                digit_str = names[class_id] if isinstance(names, list) else names.get(class_id, str(class_id))
                            else:
                                digit_str = str(class_id)
                            detected_digits.append((int(x1_pad + dx1), digit_str, d_conf))
                            
                        if detected_digits:
                            final_number_str, is_repaired, log = repair_number(detected_digits)
                            avg_conf = sum([d[2] for d in detected_digits]) / len(detected_digits)
                            
                            if len(final_number_str) == 8 and avg_conf > 0.5:
                                score_boost = 1.0
                                is_valid = validate_wagon_number(final_number_str)
                                
                                if is_valid:
                                    score_boost = 3.0
                                    if is_repaired: score_boost = 2.0
                                    text_color = (0, 255, 0)
                                else:
                                    score_boost = 0.5
                                    text_color = (0, 0, 255)
                                
                                track_scores[track_id][final_number_str] += (avg_conf * score_boost)
                                label = final_number_str
                                if is_repaired: label += "°"
                                cv2.putText(frame, label, (x1_orig, y1_orig - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                
                if is_stable:
                    cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 3)
                    cv2.putText(frame, f"ID:{track_id}", 
                                (x1_orig, y1_orig - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        dead_tracks = []
        for tid, last_frame in last_seen_frame.items():
            if frame_count - last_frame > FRAMES_TO_FORGET:
                process_finished_track(tid, track_scores.get(tid), track_lifespan.get(tid))
                dead_tracks.append(tid)
        
        for tid in dead_tracks:
            if tid in last_seen_frame: del last_seen_frame[tid]
            if tid in track_scores: del track_scores[tid]
            if tid in track_lifespan: del track_lifespan[tid]
        
        display_frame = frame
        if width > 1920: display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Wagon Number Recognition RealTime", display_frame)
        if out: out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    if out: out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Work completed.")

if __name__ == "__main__":
    run_realtime()
