"""
Detection algorithms for human and color detection
Contains YOLO human detection and red color detection
"""

import cv2
import numpy as np
from src.config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT,
    RED_DETECTION_THRESHOLD, YOLO_CONFIDENCE_THRESHOLD,
    LOWER_RED1, UPPER_RED1, LOWER_RED2, UPPER_RED2
)
import src.shared_state as state


def process_human_detection(frame):
    """Process human detection and return processed frame with detection info"""
    try:
        output_frame = frame.copy()
        
        # YOLOv8 Human Detection
        results = state.yolo_model(frame, verbose=False)

        detected = False
        human_boxes = []

        # Process YOLO results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Check if it's a person (class_id = 0 in COCO dataset)
                    if class_id == 0 and confidence > YOLO_CONFIDENCE_THRESHOLD:
                        detected = True

                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        human_boxes.append((x1, y1, x2, y2, confidence))

                        # Draw bounding box
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Calculate center of bounding box
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # Draw center point
                        cv2.circle(output_frame, (center_x, center_y), 8, (0, 255, 0), cv2.FILLED)

                        # Add label
                        confidence_percentage = confidence * 100
                        label = f"Human: {confidence_percentage:.0f}%"

                        cv2.putText(output_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Process detailed body part detection if human detected
        if detected:
            output_frame = process_body_parts(frame, output_frame)

        return output_frame, detected, len(human_boxes)
    
    except Exception as e:
        print(f"Error in human detection: {e}")
        return frame, False, 0


def process_body_parts(original_frame, output_frame):
    """Process MediaPipe body part detection"""
    try:
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        
        # Process with pose detection
        pose_results = state.pose.process(rgb_frame)

        # Process with hand detection
        hands_results = state.hands.process(rgb_frame)

        # Draw hands if detected
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                state.mp_drawing.draw_landmarks(
                    output_frame,
                    hand_landmarks,
                    state.mp_hands.HAND_CONNECTIONS,
                    state.mp_drawing_styles.get_default_hand_landmarks_style(),
                    state.mp_drawing_styles.get_default_hand_connections_style()
                )

        # Draw pose landmarks if detected
        if pose_results.pose_landmarks:
            state.mp_drawing.draw_landmarks(
                output_frame,
                pose_results.pose_landmarks,
                state.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=state.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        return output_frame
    
    except Exception as e:
        print(f"Error in body part detection: {e}")
        return output_frame


def detect_red_in_roi(img):
    """Detect red color specifically in ROI area"""
    try:
        if img is None:
            return False, None, None, 0
        
        bgr_img = img
        
        # Extract ROI
        roi = bgr_img[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH]
        
        # Detect red color in ROI
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array(LOWER_RED1)
        upper_red1 = np.array(UPPER_RED1)
        lower_red2 = np.array(LOWER_RED2)
        upper_red2 = np.array(UPPER_RED2)

        mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        mask_roi = cv2.bitwise_or(mask1, mask2)
        
        # Check if red color is detected
        red_detected = np.sum(mask_roi) > RED_DETECTION_THRESHOLD 
        pixel_count = np.sum(mask_roi > 0)
        
        # Create full-size mask for visualization
        full_mask = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=np.uint8)
        full_mask[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH] = mask_roi
        
        return red_detected, mask_roi, full_mask, pixel_count
    
    except Exception as e:
        print(f"Error in red detection ROI: {e}")
        return False, None, None, 0


def detect_red_color(img):
    """Detect red color using OpenCV for full image"""
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array(LOWER_RED1)
        upper_red1 = np.array(UPPER_RED1)
        lower_red2 = np.array(LOWER_RED2)
        upper_red2 = np.array(UPPER_RED2)

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        result = cv2.bitwise_and(img, img, mask=mask)
        return mask, result
    
    except Exception as e:
        print(f"Error in red color detection: {e}")
        return None, None