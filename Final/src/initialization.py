"""
System initialization functions
Handles pygame, Tello, and AI models setup
"""

import pygame
import os
import time
from djitellopy import Tello
from ultralytics import YOLO
import mediapipe as mp

from src.config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, SPEED, SCREENSHOTS_DIR, 
    RECORDINGS_DIR, MEDIAPIPE_MIN_DETECTION_CONFIDENCE, 
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE
)
import src.shared_state as state


def initialize_pygame():
    """Initialize pygame and create window"""
    try:
        pygame.init()
        pygame.display.set_caption("Tello video stream")
        state.screen = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])
        
        # Initialize joystick
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            state.joystick = pygame.joystick.Joystick(0)
            state.joystick.init()
            print(f"Joystick initialized: {state.joystick.get_name()}")
        else:
            print("No joystick detected - using keyboard only")
        
        # Create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // 60)  # 60 FPS timer
        return True
    except Exception as e:
        print(f"Failed to initialize pygame: {e}")
        return False


def initialize_tello():
    """Initialize Tello drone connection"""
    try:
        state.tello = Tello()
        state.tello.connect()
        state.tello.set_speed(SPEED)
        state.battery_level = state.tello.get_battery()
        print(f"Battery: {state.battery_level}%")
        
        # Start video stream
        state.tello.streamoff()
        time.sleep(0.5)
        state.tello.streamon()
        
        return True
    except Exception as e:
        print(f"Failed to connect to Tello: {e}")
        return False


def initialize_ai_models():
    """Initialize AI models for detection"""
    try:
        # Load YOLOv8 model for human detection
        print("Loading YOLOv8 model...")
        state.yolo_model = YOLO('yolov8n.pt')
        
        # Mediapipe modules for body part detection
        state.mp_drawing = mp.solutions.drawing_utils
        state.mp_drawing_styles = mp.solutions.drawing_styles
        state.mp_pose = mp.solutions.pose
        state.mp_hands = mp.solutions.hands
        
        # Create MediaPipe models
        state.pose = state.mp_pose.Pose(
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True
        )
        
        state.hands = state.mp_hands.Hands(
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
            max_num_hands=2
        )
        
        # Initialize current_detection with default values
        state.current_detection = {
            'red_detected': False,
            'mask': None,
            'result': None,
            'roi_mask': None,
            'full_roi_mask': None,
            'pixel_count': 0
        }
        
        return True
    except Exception as e:
        print(f"Failed to initialize AI models: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    try:
        for directory in [SCREENSHOTS_DIR, RECORDINGS_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
        return True
    except Exception as e:
        print(f"Failed to create directories: {e}")
        return False


def initialize_all_systems():
    """Initialize all systems in correct order"""
    print("Initializing systems...")
    
    if not create_directories():
        return False
        
    if not initialize_pygame():
        return False
        
    if not initialize_tello():
        return False
        
    if not initialize_ai_models():
        return False
    
    print("All systems initialized successfully!")
    return True


def cleanup_systems():
    """Cleanup all systems"""
    try:
        if state.tello:
            state.tello.streamoff()
            state.tello.end()
        
        if state.pose:
            state.pose.close()
            
        if state.hands:
            state.hands.close()
            
        pygame.quit()
        print("Systems cleaned up successfully")
    except Exception as e:
        print(f"Error during cleanup: {e}")