import streamlit as st
import cv2
import numpy as np
import time
import torch
import win32gui
from streamlit_modal import Modal
import win32ui
import win32con
import threading
import multiprocessing as mp
from reportlab.lib.units import mm
from datetime import datetime, date
from ultralytics import YOLO
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
import psycopg2
from psycopg2 import Error
import base64
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
from PIL import Image as PILImage
import queue
import xu_ly_anh.utils_rotate as utils_rotate
import xu_ly_anh.helper as helper

# Bản đồ ký tự cho nhận diện biển số
char_map = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H',
    18: 'K', 19: 'L', 20: 'M', 21: 'N', 22: 'P', 23: 'R', 24: 'S', 25: 'T',
    26: 'U', 27: 'V', 28: 'X', 29: 'Y', 30: 'Z'
}

# Thiết lập thư mục
VIOLATION_DIR = os.path.join(os.getcwd(), "violations")
if not os.path.exists(VIOLATION_DIR):
    os.makedirs(VIOLATION_DIR)

# Biến toàn cục cho thread và process
capture_running = False
capture_running_lock = threading.Lock()
latest_frame = None
frame_lock = threading.Lock()

# Khởi tạo trạng thái trong session_state
if "vehicle_positions" not in st.session_state:
    st.session_state.vehicle_positions = {}
if "captured_entities" not in st.session_state:
    st.session_state.captured_entities = set()
if "context_mapping" not in st.session_state:
    st.session_state.context_mapping = {}
if "processing_initialized" not in st.session_state:
    st.session_state.processing_initialized = False
if "last_violation_check" not in st.session_state:
    st.session_state.last_violation_check = 0

# Hàm tính khoảng cách
def calculate_distance(center1, center2):
    x1, y1 = center1
    x2, y2 = center2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Hàm cơ sở dữ liệu
def init_db_connection():
    try:
        connection = psycopg2.connect(
            host="localhost",
            port="5433",
            database="traffic_violations",
            user="postgres",
            password="mysecretpassword"
        )
        cursor = connection.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS violations (
            id SERIAL PRIMARY KEY,
            id_phuong_tien INTEGER NOT NULL,
            thoi_gian_vi_pham TIMESTAMP NOT NULL,
            anh_bang_chung VARCHAR(255) NOT NULL,
            bien_so_xe VARCHAR(20),
            loai_vi_pham VARCHAR(50)
        );
        """
        cursor.execute(create_table_query)
        connection.commit()
        return connection, cursor
    except (Exception, Error):
        return None, None

def save_violation_to_db(cursor, connection, violation):
    try:
        insert_query = """
        INSERT INTO violations (id_phuong_tien, thoi_gian_vi_pham, anh_bang_chung, bien_so_xe, loai_vi_pham)
        VALUES (%s, %s, %s, %s, %s);
        """
        cursor.execute(insert_query, (
            violation["id_phuong_tien"],
            violation["thoi_gian_vi_pham"],
            violation["anh_bang_chung"],
            violation["bien_so_xe"],
            violation["loai_vi_pham"]
        ))
        connection.commit()
    except (Exception, Error):
        connection.rollback()

# Hàm chụp cửa sổ
def capture_window(window_title):
    try:
        hwnd = win32gui.FindWindow(None, window_title)
        if not hwnd:
            return None, "Không tìm thấy cửa sổ scrcpy với tiêu đề: " + window_title
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        width = right - left
        height = bottom - top
        region_left, region_top, region_right, region_bottom = 0, 128, 414, 870
        capture_width = region_right - region_left
        capture_height = region_bottom - region_top
        if capture_width <= 0 or capture_height <= 0:
            return None, "Kích thước vùng chụp không hợp lệ."
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        compatible_dc = mfc_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, capture_width, capture_height)
        compatible_dc.SelectObject(bitmap)
        compatible_dc.BitBlt((0, 0), (capture_width, capture_height), mfc_dc, (region_left, region_top), win32con.SRCCOPY)
        bitmap_bits = bitmap.GetBitmapBits(True)
        img = np.frombuffer(bitmap_bits, dtype=np.uint8).reshape((capture_height, capture_width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        win32gui.DeleteObject(bitmap.GetHandle())
        compatible_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        return img, None
    except Exception as e:
        return None, f"Lỗi khi chụp: {str(e)}"

def capture_loop(window_title, frame_queue, stop_event):
    global capture_running, latest_frame
    while True:
        with capture_running_lock:
            if not capture_running or stop_event.is_set():
                break
        frame, error = capture_window(window_title)
        if error:
            with frame_lock:
                latest_frame = None
            time.sleep(0.005)
            continue
        with frame_lock:
            latest_frame = frame
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass
        time.sleep(0.005)

def point_above_line(x, y, line_start, line_end):
    x1, y1 = line_start
    x2, y2 = line_end
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A * x + B * y + C > 0

def save_violation_image(frame, box, violation, track_id, captured_entities, yolo_LP_detect, yolo_license_plate):
    if track_id in captured_entities:
        return None, "unknown"
    
    x1, y1, x2, y2 = map(int, box)
    violation_frame = frame.copy()
    color = (0, 0, 255)
    cv2.rectangle(violation_frame, (x1, y1), (x2, y2), color, 1)
    cv2.putText(violation_frame, f"ID: {track_id}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    crop_img = violation_frame[y1:y2, x1:x2]
    plates = yolo_LP_detect(crop_img, imgsz=640)
    license_plate = "unknown"
    
    for result in plates[0].boxes:
        px1, py1, px2, py2 = result.xyxy[0].tolist()
        px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)
        plate_crop = crop_img[py1:py2, px1:px2]
        cv2.rectangle(violation_frame, (x1 + px1, y1 + py1), (x1 + px2, y1 + py2), (0, 0, 255), 2)
        flag = 0
        for cc in range(0, 2):
            for ct in range(0, 2):
                processed_img = utils_rotate.deskew(plate_crop, cc, ct)
                results = yolo_license_plate(processed_img, imgsz=640)
                bb_list = map_yolo11_results(results)
                lp = read_plate_with_bb(bb_list)
                if lp != "unknown":
                    license_plate = lp
                    cv2.putText(violation_frame, lp, (x1, y2 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    flag = 1
                    break
            if flag == 1:
                break
    
    timestamp = violation["thoi_gian_vi_pham"].replace(":", "-")
    filepath = os.path.join(VIOLATION_DIR, f"violation_{track_id}_{timestamp}.jpg")
    cv2.imwrite(filepath, violation_frame)
    captured_entities.add(track_id)
    
    return filepath, license_plate

def nms_boxes(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, iou_threshold)
    return [i for i in range(len(boxes)) if i in indices]

def detect_traffic_light_and_vehicles(frame, violating_track_ids, 
                                      context_mapping, traffic_light_model, vehicle_model, 
                                      deepsort_tracker, device, VEHICLE_CLASSES):
    annotated_frame = frame.copy()
    traffic_lights = []
    tracked_objects = []
    traffic_results = traffic_light_model(frame, verbose=False, device=device)
    for result in traffic_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if conf > 0.5:
                x_center = (x1 + x2) // 2
                traffic_lights.append({"label": traffic_light_model.names[cls], "x_center": x_center, "y1": y1, "y2": y2})
                if cls == 0:
                    color = (0, 255, 0)
                elif cls == 1:
                    color = (0, 0, 255)
                elif cls == 2:
                    color = (0, 255, 255)
                else:
                    color = (128, 128, 128)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
    vehicle_results = vehicle_model(frame, classes=VEHICLE_CLASSES, verbose=False, device=device)
    detections = []
    boxes = []
    scores = []
    for result in vehicle_results:
        boxes.extend([list(map(int, box.xyxy[0])) for box in result.boxes])
        scores.extend([float(box.conf[0]) for box in result.boxes])
    if boxes:
        indices = nms_boxes([[x1, y1, x2, y2] for x1, y1, x2, y2 in boxes], scores)
        for idx in indices:
            x1, y1, x2, y2 = boxes[idx]
            conf = scores[idx]
            cls = int(vehicle_results[0].boxes.cls[idx])
            if conf > 0.5:
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
 
    current_time = time.time()
    DISTANCE_THRESHOLD = 150
    TIME_THRESHOLD = 5.0
    
    tracks = deepsort_tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cls = track.det_class
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        area = (x2 - x1) * (y2 - y1)
        color = (255, 0, 0) if track_id in violating_track_ids else (0, 255, 0)
        label = f"ID: {track_id} (Violation)" if track_id in violating_track_ids else f"ID: {track_id}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        tracked_objects.append({
            'id': [track_id],
            'xyxy': [x1, y1, x2, y2],
            'conf': [track.det_conf],
            'cls': [cls]
        })
        if track_id not in context_mapping:
            context_mapping[track_id] = {
                'center': (center_x, center_y),
                'timestamp': current_time,
                'area': area,
                'cls': cls
            }
        else:
            prev_context = context_mapping[track_id]
            prev_center = prev_context['center']
            prev_time = prev_context['timestamp']
            prev_area = prev_context['area']
            distance = calculate_distance((center_x, center_y), prev_center)
            time_diff = current_time - prev_time
            area_ratio = min(area, prev_area) / max(area, prev_area)
            if distance < DISTANCE_THRESHOLD and time_diff < TIME_THRESHOLD and area_ratio > 0.7:
                context_mapping[track_id] = {
                    'center': (center_x, center_y),
                    'timestamp': current_time,
                    'area': area,
                    'cls': cls
                }
    context_mapping = {k: v for k, v in context_mapping.items() if v['timestamp'] > current_time - TIME_THRESHOLD}
    frame_height, frame_width = frame.shape[:2]
    line_middle = [(150, 265), (650, 260)]
    line_parallel = [(160, 80), (530, 80)]
    line_left = [(130, 10), (130, 130)]
    line_right = [(620, 40), (620, 120)]
    cv2.line(annotated_frame, line_middle[0], line_middle[1], (0, 0, 255), 1)
    cv2.line(annotated_frame, line_parallel[0], line_parallel[1], (255, 0, 0), 1)
    cv2.line(annotated_frame, line_left[0], line_left[1], (0, 255, 0), 1)
    cv2.line(annotated_frame, line_right[0], line_right[1], (255, 255, 0), 1)
    lines = (line_middle, line_left, line_right, line_parallel)
    return annotated_frame, traffic_lights, tracked_objects, lines, context_mapping

def map_yolo11_results(results):
    bb_list = []
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].tolist()
        cls_id = int(result.cls.item())
        conf = result.conf.item()
        if conf >= 0.50:
            bb_list.append([x1, y1, x2, y2, char_map.get(cls_id, '?')])
    return bb_list

def read_plate_with_bb(bb_list):
    LP_type = "1"
    if len(bb_list) == 0 or len(bb_list) < 7 or len(bb_list) > 10:
        return "unknown"
    
    center_list = []
    y_sum = 0
    for bb in bb_list:
        x_c = (bb[0] + bb[2]) / 2
        y_c = (bb[1] + bb[3]) / 2
        y_sum += y_c
        center_list.append([x_c, y_c, bb[4]])
    
    l_point = center_list[0]
    r_point = center_list[0]
    for cp in center_list:
        if cp[0] < l_point[0]:
            l_point = cp
        if cp[0] > r_point[0]:
            r_point = cp
    
    for ct in center_list:
        if l_point[0] != r_point[0]:
            if not helper.check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]):
                LP_type = "2"
    
    y_mean = int(y_sum / len(bb_list))
    
    license_plate = ""
    if LP_type == "2":
        line_1 = []
        line_2 = []
        for c in center_list:
            if int(c[1]) > y_mean:
                line_2.append(c)
            else:
                line_1.append(c)
        for l1 in sorted(line_1, key=lambda x: x[0]):
            license_plate += str(l1[2])
        license_plate += "-"
        for l2 in sorted(line_2, key=lambda x: x[0]):
            license_plate += str(l2[2])
    else:
        for l in sorted(center_list, key=lambda x: x[0]):
            license_plate += str(l[2])
    
    return license_plate

def detect_violation(frame, traffic_lights, tracked_objects, lines, 
                     context_mapping, vehicle_positions, 
                     captured_entities, yolo_LP_detect, yolo_license_plate):
    violations = []
    violating_track_ids = set()
    frame_height, frame_width = frame.shape[:2]
    
    connection, cursor = init_db_connection()
    if connection is None or cursor is None:
        return violations, violating_track_ids

    line_middle, line_left, line_right, line_parallel = lines
    
    try:
        traffic_light = next((tl for tl in traffic_lights if tl['label'] == 'Red'), None)
        
        # Duyệt qua các đối tượng được theo dõi
        for obj in tracked_objects:
            track_id = int(obj['id'][0])  
            x1, y1, x2, y2 = map(int, obj['xyxy'])  
            cls = int(obj['cls'][0])  
            center_x = (x1 + x2) // 2  
            center_y = (y1 + y2) // 2 
            current_time = time.time()  
            
            if track_id not in vehicle_positions:
                vehicle_positions[track_id] = {
                    'positions': [], 
                    'timestamp_middle': None,  
                    'light_state': None, 
                    'has_stopped': False,  
                    'route': None, 
                    'crossed_middle': False, 
                    'cls': cls, 
                    'area_history': []  
                }
            
            entity_data = vehicle_positions[track_id]
            entity_data['positions'].append((center_x, center_y, current_time))
            entity_data['area_history'].append((x2 - x1) * (y2 - y1))
            
            if len(entity_data['positions']) > 120:
                entity_data['positions'].pop(0)
                entity_data['area_history'].pop(0)
            
            if not entity_data['crossed_middle']:
                above_middle = point_above_line(center_x, center_y, line_middle[0], line_middle[1])
                if len(entity_data['positions']) >= 2:
                    prev_center_x, prev_center_y, _ = entity_data['positions'][-2]
                    prev_above_middle = point_above_line(prev_center_x, prev_center_y, line_middle[0], line_middle[1])
                    if prev_above_middle != above_middle:
                        entity_data['crossed_middle'] = True
                        entity_data['timestamp_middle'] = current_time
                        entity_data['light_state'] = 'Red' if traffic_light else 'Not Red'
                        entity_data['has_stopped'] = False
                        entity_data['route'] = None
            
            if entity_data['crossed_middle'] and not entity_data['has_stopped']:
                if len(entity_data['positions']) >= 2:
                    first_pos = entity_data['positions'][0]
                    last_pos = entity_data['positions'][-1]
                    time_diff = last_pos[2] - first_pos[2]
                    x_diff = abs(last_pos[0] - first_pos[0])
                    y_diff = abs(last_pos[1] - first_pos[1])
                    if time_diff >= 2.0 and x_diff < 10 and y_diff < 10:
                        entity_data['has_stopped'] = True
            
            if entity_data['crossed_middle'] and entity_data['route'] is None:
                above_parallel = point_above_line(center_x, center_y, line_parallel[0], line_parallel[1])
                above_left = point_above_line(center_x, center_y, line_left[0], line_left[1])
                above_right = point_above_line(center_x, center_y, line_right[0], line_right[1])
                if len(entity_data['positions']) >= 2:
                    prev_center_x, prev_center_y, _ = entity_data['positions'][-2]
                    prev_above_parallel = point_above_line(prev_center_x, prev_center_y, line_parallel[0], line_parallel[1])
                    prev_above_left = point_above_line(prev_center_x, prev_center_y, line_left[0], line_left[1])
                    prev_above_right = point_above_line(prev_center_x, prev_center_y, line_right[0], line_right[1])
                    if prev_above_parallel != above_parallel:
                        entity_data['route'] = 'straight'
                    elif prev_above_left != above_left:
                        entity_data['route'] = 'left'
                    elif prev_above_right != above_right:
                        entity_data['route'] = 'right'
            
            if entity_data['crossed_middle'] and entity_data['route'] is not None and track_id not in captured_entities:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                violation = None
                if cls in [2, 3, 5, 7]:  
                    if (entity_data['route'] in ['straight', 'left']) and entity_data['light_state'] == 'Red' and not entity_data['has_stopped']:
                        violation_type = "Vượt đèn đỏ đi thẳng" if entity_data['route'] == 'straight' else "Vượt đèn đỏ rẽ trái"
                        filepath, license_plate = save_violation_image(
                            frame, [x1, y1, x2, y2], {"thoi_gian_vi_pham": timestamp}, 
                            track_id, captured_entities, yolo_LP_detect, yolo_license_plate
                        )
                        violation = {
                            "id_phuong_tien": track_id,
                            "thoi_gian_vi_pham": timestamp,
                            "anh_bang_chung": filepath,
                            "bien_so_xe": license_plate,
                            "loai_vi_pham": violation_type
                        }
                    elif entity_data['route'] == 'right':
                        continue  
                    if violation and violation["anh_bang_chung"]:
                        save_violation_to_db(cursor, connection, violation)
                        violations.append(violation)
                        violating_track_ids.add(track_id)
                        captured_entities.add(track_id)
    except Exception as e:
        if connection:
            connection.rollback()
        print(f"Lỗi trong quá trình phát hiện vi phạm: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
    
    return violations, violating_track_ids

def create_pdf_report(record, full_img_path):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    try:
        pdfmetrics.registerFont(TTFont('DejaVuSansExtraLight','dejavu-sans.extralight.ttf'))
        styles['Title'].fontName = 'DejaVuSansExtraLight'
        styles['Normal'].fontName = 'DejaVuSansExtraLight'
    except Exception as e:
        st.warning(f"Không thể tải font DejaVuSansExtraLight: {str(e)}. Sử dụng font mặc định (Times-Roman).")
        styles['Title'].fontName = 'Times-Roman'
        styles['Normal'].fontName = 'Times-Roman'
    elements = []
    elements.append(Paragraph(f"Báo cáo vi phạm giao thông - ID: {record[0]}", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"ID phương tiện: {record[1]}", styles['Normal']))
    elements.append(Paragraph(f"Thời gian vi phạm: {record[2].strftime('%Y-%m-%d %H:%M:%S') if record[2] else 'N/A'}", styles['Normal']))
    elements.append(Paragraph(f"Biển số xe: {record[4] if record[4] else 'Không có'}", styles['Normal']))
    elements.append(Paragraph(f"Loại vi phạm: {record[5] if record[5] else 'Không xác định'}", styles['Normal']))
    elements.append(Spacer(1, 12))
    if os.path.exists(full_img_path):
        try:
            img = PILImage.open(full_img_path)
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            target_width = 150 * mm
            target_height = target_width * aspect
            if target_height > 100 * mm:
                target_height = 100 * mm
                target_width = target_height / aspect
            elements.append(Image(full_img_path, width=target_width, height=target_height))
        except Exception as e:
            elements.append(Paragraph(f"Lỗi khi chèn ảnh: {str(e)}", styles['Normal']))
    else:
        elements.append(Paragraph("Không tìm thấy ảnh bằng chứng", styles['Normal']))
    doc.build(elements)
    buffer.seek(0)
    return buffer

st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css" rel="stylesheet">
    <style>
        .modal-content {
            background-color: #fff;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            padding-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 95%;
            width: 100%;
            margin-left: auto;
            margin-right: auto;
            max-height: 350px;
            overflow-y: auto;
        }
        .modal-content h4 {
            font-size: 20px;
            color: #2c3e50;
            margin: 0 0 10px;
        }
        .btn {
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            font-weight: 600;
        }
        .btn-primary {
            background-color: #0d6efd;
            color: white;
        }
        .btn-primary:hover {
            background-color: #0056b3;
        }
        .btn-secondary {
            background-color: #6c757d;
            color: white;
        }
        .btn-secondary:hover {
            background-color: #5a6268;
        }
        .modal-content p {
            max-width: 80%;
            margin-left: auto;
            margin-right: auto;
        }
        .modal-content img {
            max-width: 80%;
            margin-left: auto;
            margin-right: auto;
            display: block;
        }
        .modal-content div {
            max-width: 80%;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
""", unsafe_allow_html=True)

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception:
        return None

def initialize_processing(frame_queue, result_queue, stop_event):
    global capture_running
    window_title = "21081111RG"
    frame, error = capture_window(window_title)
    if error:
        return False, error
    with capture_running_lock:
        if capture_running:
            return True, None
        capture_running = True
    capture_thread = threading.Thread(
        target=capture_loop,
        args=(window_title, frame_queue, stop_event),
        daemon=True
    )
    capture_thread.start()
    detection_process = mp.Process(
        target=process_detection,
        args=(frame_queue, result_queue, stop_event),
        daemon=True
    )
    detection_process.start()
    st.session_state.capture_thread = capture_thread
    st.session_state.detection_process = detection_process
    return True, None

def stop_processing():
    global capture_running
    with capture_running_lock:
        capture_running = False
    if st.session_state.get('stop_event'):
        st.session_state.stop_event.set()
    if st.session_state.get('capture_thread'):
        st.session_state.capture_thread.join(timeout=1.0)
        st.session_state.capture_thread = None
    if st.session_state.get('detection_process'):
        st.session_state.detection_process.terminate()
        st.session_state.detection_process.join()
        st.session_state.detection_process = None
    st.session_state.frame_queue = None
    st.session_state.result_queue = None
    st.session_state.stop_event = None
    st.session_state.processing_initialized = False

def show_camera():
    st.subheader("🎥 Camera giám sát")
    info_placeholder = st.empty()
    address_text = "📍 **Địa chỉ**: GT_Ngã 4 đèn đỏ Sân vận động_4 - Thành phố Điện Biên Phủ"
    if not st.session_state.processing_initialized:
        st.error("Hệ thống chưa được khởi tạo. Vui lòng kiểm tra cửa sổ scrcpy.")
        return
    st.success("Đã tìm thấy cửa sổ scrcpy. Hiển thị luồng video...")
    video_placeholder = st.empty()
    while True:
        current_time = datetime.now().strftime('%I:%M:%S %p, %A, %d/%m/%Y')
        info_placeholder.markdown(f"🕒 **Thời gian hiện tại:** {current_time}<br>{address_text}", unsafe_allow_html=True)
        try:
            result = st.session_state.result_queue.get_nowait()
            frame_rgb = result['frame']
            violations = result['violations']
            if violations:
                st.warning(f"Phát hiện vi phạm: {len(violations)} trường hợp vượt đèn đỏ!")
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        except queue.Empty:
            pass
        time.sleep(0.01)

def show_history():
    st.subheader("📋 Lịch sử vi phạm")
    modal = Modal("📋 Báo cáo vi phạm", key="bao_cao_popup")
    if "selected_record" not in st.session_state:
        st.session_state.selected_record = None
        st.session_state.full_img_path = None
    connection, cursor = init_db_connection()
    if connection is None or cursor is None:
        st.error("Không thể hiển thị lịch sử vi phạm do lỗi kết nối cơ sở dữ liệu.")
        return
    try:
        current_time = time.time()
        if current_time - st.session_state.last_violation_check > 1.0:
            st.session_state.last_violation_check = current_time
            if st.session_state.get('result_queue'):
                try:
                    result = st.session_state.result_queue.get_nowait()
                    violations = result['violations']
                    if violations:
                        st.warning(f"Phát hiện vi phạm mới: {len(violations)} trường hợp vượt đèn đỏ!")
                except queue.Empty:
                    pass
        st.subheader("🔍 Tìm kiếm vi phạm")
        with st.form(key="search_form"):
            col1, col2 = st.columns(2)
            with col1:
                search_id = st.text_input("ID phương tiện", "")
                search_plate = st.text_input("Biển số xe", "")
            with col2:
                start_date = st.date_input("Từ ngày", value=None, format="YYYY-MM-DD")
                end_date = st.date_input("Đến ngày", value=None, format="YYYY-MM-DD")
            submit_button = st.form_submit_button("Tìm kiếm")
        query_conditions = []
        query_params = []
        if search_id:
            query_conditions.append("id_phuong_tien = %s")
            query_params.append(int(search_id) if search_id.isdigit() else 0)
        if search_plate:
            query_conditions.append("bien_so_xe ILIKE %s")
            query_params.append(f"%{search_plate}%")
        if start_date:
            query_conditions.append("thoi_gian_vi_pham >= %s")
            query_params.append(datetime.combine(start_date, datetime.min.time()))
        if start_date and not end_date:
            end_date = date.today()
            query_conditions.append("thoi_gian_vi_pham <= %s")
            query_params.append(datetime.combine(end_date, datetime.max.time()))
        elif end_date:
            query_conditions.append("thoi_gian_vi_pham <= %s")
            query_params.append(datetime.combine(end_date, datetime.max.time()))
        select_query = """
        SELECT id, id_phuong_tien, thoi_gian_vi_pham, anh_bang_chung, bien_so_xe, loai_vi_pham
        FROM violations
        """
        if query_conditions:
            select_query += " WHERE " + " AND ".join(query_conditions)
        select_query += " ORDER BY thoi_gian_vi_pham DESC"
        count_query = f"SELECT COUNT(*) FROM ({select_query}) AS subquery"
        cursor.execute(count_query, query_params)
        total_records = cursor.fetchone()[0]
        st.subheader("📄 Phân trang")
        records_per_page = st.selectbox("Số vi phạm mỗi trang", [10, 20, 50], index=0)
        total_pages = (total_records + records_per_page - 1) // records_per_page
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("Trang trước") and st.session_state.current_page > 1:
                st.session_state.current_page -= 1
        with col2:
            st.write(f"Trang {st.session_state.current_page} / {total_pages}")
        with col3:
            if st.button("Trang sau") and st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
        offset = (st.session_state.current_page - 1) * records_per_page
        paginated_query = f"{select_query} LIMIT %s OFFSET %s"
        cursor.execute(paginated_query, query_params + [records_per_page, offset])
        records = cursor.fetchall()
        if records:
            table_container = st.container()
            with table_container:
                for record in records:
                    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
                    with col1:
                        st.markdown(f"**ID:** {record[0]}")
                        st.markdown(f"**ID phương tiện:** {record[1]}")
                    with col2:
                        st.markdown(f"**Thời gian:** {record[2].strftime('%Y-%m-%d %H:%M:%S') if record[2] else 'N/A'}")
                        st.markdown(f"**Biển số xe:** {record[4] if record[4] else 'Không có'}")
                        st.markdown(f"**Loại vi phạm:** {record[5] if record[5] else 'Không xác định'}")
                    with col3:
                        img_path = record[3]
                        img_path = img_path.replace('/', os.sep)
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        full_img_path = os.path.normpath(os.path.join(current_dir, img_path))
                        if os.path.exists(full_img_path):
                            try:
                                img_data = cv2.imread(full_img_path)
                                if img_data is not None:
                                    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                                    img_data = cv2.resize(img_data, (200, 150))
                                    st.image(img_data, channels="RGB", use_container_width=True)
                                else:
                                    st.error("Không thể đọc ảnh")
                            except Exception as e:
                                st.error(f"Lỗi khi đọc ảnh: {str(e)}")
                        else:
                            st.error("Không tìm thấy ảnh")
                    with col4:
                        if st.button("Xuất báo cáo", key=f"report_key_{record[0]}"):
                            st.session_state.selected_record = record
                            st.session_state.full_img_path = full_img_path
                            modal.open()
        else:
            st.write("Không tìm thấy vi phạm nào khớp với tiêu chí tìm kiếm.")
        if modal.is_open():
            with modal.container():
                record = st.session_state.selected_record
                full_img_path = st.session_state.full_img_path
                vi_pham = record[5] if record[5] else "Không xác định"
                phuong_tien = record[1]
                bien_so = str(record[4]) if record[4] else "Không có"
                thoi_gian = record[2].strftime('%Y-%m-%d %H:%M:%S') if record[2] else "N/A"
                img_tag = '<p style="color:red;">❌ Không tải được ảnh!</p>'
                if os.path.exists(full_img_path):
                    try:
                        hinh_base64 = image_to_base64(full_img_path)
                        img_tag = f'<img src="data:image/jpeg;base64,{hinh_base64}" style="width:100%; border-radius:8px; margin-top: 12px; box-shadow: 0 0 10px rgba(0,0,0,0.1);"/>'
                    except Exception as e:
                        img_tag = f'<p style="color:red;">❌ Lỗi: {str(e)}</p>'
                try:
                    pdf_buffer = create_pdf_report(record, full_img_path)
                    pdf_data = pdf_buffer.getvalue()
                    b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
                    pdf_href = f'data:application/pdf;base64,{b64_pdf}'
                except Exception as e:
                    st.error(f"Lỗi khi tạo PDF: {str(e)}")
                    pdf_href = "#"
                st.markdown(f"""
                    <div class="modal-content">
                        <h4><i class="bi bi-flag-fill"></i> Báo cáo vi phạm - ID: <span style="color:#d63031;">{record[0]}</span></h4>
                        <p><strong><i class="bi bi-exclamation-triangle-fill"></i> Vi phạm:</strong> {vi_pham}</p>
                        <p><strong><i class="bi bi-truck"></i> Phương tiện:</strong> {phuong_tien}</p>
                        <p><strong><i class="bi bi-car-front"></i> Biển số:</strong> {bien_so}</p>
                        <p><strong><i class="bi bi-clock"></i> Thời gian:</strong> {thoi_gian}</p>
                        {img_tag}
                        <div style="margin-top: 15px; display: flex; gap: 10px;">
                            <a href="{pdf_href}" download="violation_report_{record[0]}.pdf">
                                <button class="btn btn-primary"><i class="bi bi-download"></i> Tải PDF</button>
                            </a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    except Exception as error:
        st.error(f"Lỗi khi truy vấn lịch sử vi phạm: {error}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def process_detection(frame_queue, result_queue, stop_event):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang sử dụng thiết bị: {device}")

    traffic_light_model = YOLO("den_giao_thong.pt").to(device)
    vehicle_model = YOLO("xe_co.pt").to(device)
    yolo_LP_detect = YOLO("bien_so.pt").to(device)
    yolo_license_plate = YOLO("ki_tu_bien_so.pt").to(device)
    yolo_license_plate.conf = 0.60
    VEHICLE_CLASSES = [2, 3, 5, 7]
    deepsort_tracker = DeepSort(max_age=50, nn_budget=100, nms_max_overlap=0.3, max_cosine_distance=0.2)
    vehicle_positions = {}
    captured_entities = set()
    context_mapping = {}

    while not stop_event.is_set():
        try:
            frame = frame_queue.get_nowait()
            annotated_frame, traffic_lights, tracked_objects, lines, context_mapping = detect_traffic_light_and_vehicles(
                frame, set(), context_mapping, traffic_light_model, vehicle_model, deepsort_tracker, device, VEHICLE_CLASSES
            )
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (415, 295), interpolation=cv2.INTER_LANCZOS4)
            violations, violating_track_ids = detect_violation(
                frame, traffic_lights, tracked_objects, lines, context_mapping, vehicle_positions, captured_entities,
                yolo_LP_detect, yolo_license_plate
            )
            result_queue.put({
                'frame': frame_resized,
                'violations': violations,
                'error': None
            })
        except queue.Empty:
            time.sleep(0.001)
        time.sleep(0.01)

def main():
    st.sidebar.title("🚦 Giám sát vượt đèn đỏ")
    if not st.session_state.processing_initialized:
        st.session_state.frame_queue = mp.Queue(maxsize=20)
        st.session_state.result_queue = mp.Queue(maxsize=20)
        st.session_state.stop_event = mp.Event()
        success, error = initialize_processing(
            st.session_state.frame_queue,
            st.session_state.result_queue,
            st.session_state.stop_event
        )
        if not success:
            st.error(error)
            return
        st.session_state.processing_initialized = True

    page = st.sidebar.radio("Chọn chức năng", ["Camera", "Lịch sử"])
    if page == "Camera":
        show_camera()
    elif page == "Lịch sử":
        show_history()

if __name__ == '__main__':
    main()