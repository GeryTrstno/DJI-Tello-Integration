'''

import sys
import os
import threading
import queue
import time
import json
import base64
import cv2
import numpy as np
from datetime import datetime
from collections import deque
import subprocess
import signal
import atexit

# Web server imports
try:
    from flask import Flask, request, jsonify, send_file, send_from_directory
    from flask_socketio import SocketIO, emit
    from flask_cors import CORS
    WEB_IMPORTS_AVAILABLE = True
except ImportError:
    print("⚠️ Web server dependencies not available. Install: pip install flask flask-socketio flask-cors")
    WEB_IMPORTS_AVAILABLE = False

# Drone imports (from droneV7.py)
try:
    import pygame
    import cv2
    import numpy as np
    from djitellopy import Tello
    from ultralytics import YOLO
    import mediapipe as mp
    DRONE_IMPORTS_AVAILABLE = True
except ImportError:
    print("⚠️ Drone dependencies not available. Install required packages.")
    DRONE_IMPORTS_AVAILABLE = False

# ==================== CONFIGURATION ====================
class Config:
    # Web server config
    WEB_HOST = '127.0.0.1'
    WEB_PORT = 5000
    WEB_DEBUG = False
    
    # Drone config (from droneV7.py)
    FPS = 120
    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 480
    SPEED = ''
    THREAD_COUNT = 5
    
    # Integration config
    FRAME_SHARE_FILE = "shared_frame.jpg"
    STATUS_SHARE_FILE = "shared_status.json"
    COMMAND_SHARE_FILE = "shared_commands.json"

        # Media directories
    SCREENSHOTS_DIR = "screenshots"
    RECORDINGS_DIR = "recordings"
    MEDIA_BASE_URL = "/media"
    
# ==================== SHARED DATA MANAGEMENT ====================
class SharedDataManager:
    """Manages communication between drone system and web server"""
    
    def __init__(self):
        self.status_lock = threading.Lock()
        self.command_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        
        # Shared status data
        self.status_data = {
            'connected': False,
            'flying': False,
            'battery': 0,
            'speed': '',
            'temperature': 0,
            'height': 0,
            'humans_detected': 0,
            'fps': 0,
            'recording': False,
            'ml_detection_enabled': False,
            'auto_capture_enabled': False,
            'screenshot_count': 0,
            'flight_time': 0,
            'keyboard_enabled': False,
            'autonomous_mode': False,
            'autonomous_action': 'idle',
            'red_detected': False,
            'pixel_count': 0,
            'telemetry': {
                'pitch': 0, 'roll': 0, 'yaw': 0,
                'speed_x': 0, 'speed_y': 0, 'speed_z': 0,
                'accel_x': 0, 'accel_y': 0, 'accel_z': 0,
                'barometer': 0, 'tof': 0
            }
        }
        
        # Command queue
        self.command_queue = queue.Queue()
        
        # Current frame
        self.current_frame = None
        self.current_frame_base64 = None
        
        # Create directories
        self._create_directories()
    def _create_directories(self):
        """Create necessary directories"""
        directories = [Config.SCREENSHOTS_DIR, Config.RECORDINGS_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Directory ready: {directory}")
    
    def convert_numpy_types(self, obj):
        """Convert NumPy types to JSON serializable Python types"""
        if isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def update_status(self, status_update):
        """Update status data thread-safely with NumPy type conversion"""
        with self.status_lock:
            # Convert NumPy types before updating
            converted_update = self.convert_numpy_types(status_update)
            self.status_data.update(converted_update)
    
    def get_status(self):
        """Get current status data with NumPy type conversion"""
        with self.status_lock:
            # Convert NumPy types before returning
            return self.convert_numpy_types(self.status_data.copy())
    
    def add_command(self, command):
        """Add command to queue"""
        self.command_queue.put(command)
    
    def get_command(self):
        """Get command from queue (non-blocking)"""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None
    
    def update_frame(self, frame):
        """Update current frame"""
        with self.frame_lock:
            if frame is not None:
                self.current_frame = frame.copy()
                # Convert to base64 for web transmission
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, buffer = cv2.imencode('.jpg', frame_rgb)
                self.current_frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    def get_frame_base64(self):
        """Get current frame as base64"""
        with self.frame_lock:
            return self.current_frame_base64

# Global shared data manager
shared_data = SharedDataManager()

# ==================== DRONE SYSTEM (Modified droneV7.py) ====================
class DroneSystem:
    """
    Modified drone system from droneV7.py with web integration hooks
    Struktur asli dipertahankan, hanya ditambahkan web integration points
    """
    
    def __init__(self):
        # Original droneV7.py variables
        self.running = True
        self.threads = []
        
        # Tello objects
        self.tello = None
        self.screen = None
        self.joystick = None
        
        # AI models
        self.yolo_model = None
        self.pose = None
        self.hands = None
        
        # Drone state
        self.current_frame = None
        self.current_processed_frame = None
        self.battery_level = 0
        self.human_detected = False
        self.humans_count = 0
        self.fps = 0
        
        # Control variables
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.send_rc_control = False
        self.speed = 20
        
        # Recording
        self.recording = False
        self.video_writer = None
        
        # Detection
        self.detection_enabled = True
        self.current_detection = None
        
        # Autonomous behavior
        self.set_autonomous_behavior = False
        
        # Locks
        self.data_lock = threading.Lock()
        
        # Web integration flag
        self.web_integration_enabled = True
        
        print("🚁 Drone system initialized with web integration")
    
    def initialize_all_systems(self):
        """Initialize all drone systems"""
        try:
            if not self._initialize_pygame():
                return False
            if not self._initialize_tello():
                return False
            if not self._initialize_ai_models():
                return False
            
            print("✅ All drone systems initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize drone systems: {e}")
            return False
    
    def _initialize_pygame(self):
        """Initialize pygame (headless mode for web)"""
        try:
            if self.web_integration_enabled:
                # Headless mode - no display
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
            
            pygame.init()
            pygame.display.set_caption("Tello video stream")
            
            if not self.web_integration_enabled:
                self.screen = pygame.display.set_mode([Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT])
            
            # Initialize joystick
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"🎮 Joystick initialized: {self.joystick.get_name()}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to initialize pygame: {e}")
            return False
    
    def _initialize_tello(self):
        """Initialize Tello drone connection"""
        try:
            self.tello = Tello()
            self.tello.connect()
            self.tello.set_speed(self.speed)
            self.battery_level = self.tello.get_battery()
            print(f"🔋 Battery: {self.battery_level}%")
            
            # Start video stream
            self.tello.streamoff()
            time.sleep(0.5)
            self.tello.streamon()
            
            # Update shared data
            shared_data.update_status({
                'connected': True,
                'battery': self.battery_level,
                'speed': self.speed
            })
            
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Tello: {e}")
            shared_data.update_status({'connected': False})
            return False
    
    def _initialize_ai_models(self):
        """Initialize AI models"""
        try:
            print("🤖 Loading AI models...")
            self.yolo_model = YOLO('yolov8n.pt')
            
            # MediaPipe
            mp_pose = mp.solutions.pose
            mp_hands = mp.solutions.hands
            
            self.pose = mp_pose.Pose(
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
                model_complexity=1
            )
            
            self.hands = mp_hands.Hands(
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
                max_num_hands=2
            )
            
            print("✅ AI models loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize AI models: {e}")
            return False
    
    def start_drone_threads(self):
        """Start all drone threads"""
        print("🚀 Starting drone threads...")
        
        thread_configs = [
            ("Video Stream", self._video_stream_thread),
            ("Drone Control", self._drone_control_thread),
            ("Detection", self._detection_thread),
            ("Autonomous Behavior", self._autonomous_behavior_thread),
            ("Web Integration", self._web_integration_thread)
        ]
        
        for name, target_func in thread_configs:
            thread = threading.Thread(target=target_func, daemon=True, name=name)
            thread.start()
            self.threads.append(thread)
        
        print(f"✅ Started {len(self.threads)} drone threads")
    
    def _video_stream_thread(self):
        """Handle video capture and processing"""
        print("📹 Video stream thread started")
        
        try:
            frame_read = self.tello.get_frame_read()
            frame_times = deque(maxlen=30)
            
            while self.running:
                try:
                    if frame_read.stopped:
                        break
                    
                    frame = frame_read.frame
                    if frame is not None:
                        # Resize frame
                        frame = cv2.resize(frame, (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
                        
                        # Process detection if enabled
                        if self.detection_enabled:
                            output_frame, detected, count = self._process_human_detection(frame)
                        else:
                            output_frame = frame.copy()
                            detected = False
                            count = 0
                        
                        # Update shared data
                        with self.data_lock:
                            self.current_frame = frame.copy()
                            self.current_processed_frame = output_frame.copy()
                            self.human_detected = detected
                            self.humans_count = count
                            
                            # Calculate FPS
                            current_time = time.time()
                            frame_times.append(current_time)
                            if len(frame_times) > 1:
                                time_diff = frame_times[-1] - frame_times[0]
                                self.fps = len(frame_times) / time_diff if time_diff > 0 else 0
                        
                        # Update shared frame for web
                        shared_data.update_frame(output_frame)
                        if self.recording and self.video_writer:
                            # Pastikan frame format benar
                            frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                            self.video_writer.write(frame_bgr)
                        # Update status
                        shared_data.update_status({
                            'fps': self.fps,
                            'humans_detected': count,
                            'ml_detection_enabled': self.detection_enabled
                        })
                    
                    time.sleep(0.01)  # Control processing speed
                    
                except Exception as e:
                    print(f"❌ Video stream error: {e}")
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"❌ Critical video stream error: {e}")
        
        print("📹 Video stream thread ended")
    
    def _drone_control_thread(self):
        """Handle drone control commands"""
        print("🎮 Drone control thread started")
        
        last_battery_check = time.time()
        
        while self.running:
            try:
                # Check for web commands
                command = shared_data.get_command()
                if command:
                    self._execute_web_command(command)
                
                # Send RC control if active
                if self.send_rc_control and self.tello:
                    try:
                        self.tello.send_rc_control(
                            self.left_right_velocity,
                            self.for_back_velocity, 
                            self.up_down_velocity,
                            self.yaw_velocity
                        )
                    except Exception as e:
                        print(f"❌ RC command error: {e}")
                
                # Update battery periodically
                current_time = time.time()
                if current_time - last_battery_check >= 10:  # Every 10 seconds
                    try:
                        if self.tello:
                            self.battery_level = self.tello.get_battery()
                            shared_data.update_status({'battery': self.battery_level})
                        last_battery_check = current_time
                    except Exception as e:
                        print(f"❌ Battery check error: {e}")
                
                time.sleep(1/30)  # 30 FPS control loop
                
            except Exception as e:
                print(f"❌ Drone control error: {e}")
                time.sleep(0.1)
        
        print("🎮 Drone control thread ended")

    def _detect_red_in_roi(self, frame):
        """Detect red color in ROI for autonomous behavior - Fixed with type conversion"""
        try:
            if frame is None:
                return False, 0
            
            # Define ROI parameters
            roi_x = Config.WINDOW_WIDTH // 4
            roi_y = 5
            roi_width = Config.WINDOW_WIDTH // 2
            roi_height = Config.WINDOW_HEIGHT // 3
            
            # Extract ROI
            roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
            
            # Handle color format
            if len(frame.shape) == 3:
                roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            else:
                roi_bgr = roi
            
            # Convert to HSV
            hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            
            # Red color ranges
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([179, 255, 255])
            
            # Create masks
            mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
            mask_roi = cv2.bitwise_or(mask1, mask2)
            
            # Count red pixels
            pixel_count = np.sum(mask_roi > 0)
            
            # PERBAIKAN: Convert NumPy types to Python native types
            pixel_count = int(pixel_count)  # numpy.int64 -> int
            red_detected = pixel_count > 5000
            
            # if pixel_count > 0:
            #     print(f"🔍 Red detection: pixels={pixel_count}, detected={red_detected}")
            
            return red_detected, pixel_count
            
        except Exception as e:
            print(f"❌ Red detection error: {e}")
            return False, 0  # Python native types
        
    def _detection_thread(self):
        """Handle AI detection - Fixed version"""
        print("🤖 Detection thread started")
        
        while self.running:
            try:
                if self.current_processed_frame is not None:
                    # Get current frame safely
                    with self.data_lock:
                        if self.current_processed_frame is not None:
                            frame_copy = self.current_processed_frame.copy()
                        else:
                            frame_copy = None
                    
                    if frame_copy is not None:
                        # PERBAIKAN: Handle format warna dengan benar
                        # Jika frame dari YOLO adalah BGR, tidak perlu convert
                        # Jika frame dari YOLO adalah RGB, handle di _detect_red_in_roi
                        
                        # Perform red color detection for autonomous behavior
                        red_detected, pixel_count = self._detect_red_in_roi(frame_copy)
                        
                        # Update detection results safely
                        with self.data_lock:
                            self.current_detection = {
                                'red_detected': red_detected,
                                'pixel_count': pixel_count,
                                'timestamp': time.time()  # Tambahkan timestamp
                            }
                        
                        # Update shared data untuk web interface
                        shared_data.update_status({
                            'red_detected': red_detected,
                            'pixel_count': pixel_count
                        })
                else:
                    # Clear detection when disabled
                    with self.data_lock:
                        self.current_detection = {
                            'red_detected': False,
                            'pixel_count': 0,
                            'timestamp': time.time()
                        }
                
                time.sleep(0.03)  # Control detection frequency
                
            except Exception as e:
                print(f"❌ Detection thread error: {e}")
                time.sleep(0.1)
        
        print("🤖 Detection thread ended")

    def _autonomous_behavior_thread(self):
        """Handle autonomous behavior - Fixed version"""
        print("🤖 Autonomous behavior thread started")
        
        last_action_time = 0
        action_cooldown = 3.0  # Cooldown antara action
        
        while self.running:
            try:
                # Only run autonomous behavior if conditions are met
                if (self.set_autonomous_behavior and 
                     self.tello):
                    
                    current_time = time.time()
                    
                    # Get current detection results safely
                    with self.data_lock:
                        if self.current_detection:
                            red_detected = self.current_detection.get('red_detected', False)
                            pixel_count = self.current_detection.get('pixel_count', 0)
                            detection_time = self.current_detection.get('timestamp', 0)
                        else:
                            red_detected = False
                            pixel_count = 0
                            detection_time = 0
                    
                    # Hanya ambil action jika ada deteksi baru dan cooldown selesai
                    if (current_time - last_action_time > action_cooldown and
                        current_time - detection_time < 1.0):  # Detection tidak lebih dari 1 detik
                        
                        if red_detected:
                            print(f"🔴 Red detected in ROI! Pixel count: {pixel_count}")
                            print("🎯 Moving towards target...")
                            
                            try:
                                # PERBAIKAN: Gerakan yang lebih halus
                                self.tello.move_back(20)  # Turunkan dari 70 ke 30
                                time.sleep(2)
                                self.tello.rotate_clockwise(90)  # Turunkan dari 90 ke 45
                                time.sleep(2)
                                
                                last_action_time = current_time
                                
                                # Update status
                                shared_data.update_status({
                                    'autonomous_action': 'target_detected',
                                    'red_detected': True,
                                    'pixel_count': pixel_count
                                })
                                
                            except Exception as move_error:
                                print(f"❌ Autonomous movement error: {move_error}")
                        
                        elif current_time - last_action_time > action_cooldown * 2:
                            # Search behavior - hanya jika sudah lama tidak ada action
                            print("⚪ No red in ROI. Searching...")
                            
                            try:
                                self.tello.move_forward(20)  # Turunkan dari 30 ke 20
                                time.sleep(1)
                                
                                last_action_time = current_time
                                
                                # Update status
                                shared_data.update_status({
                                    'autonomous_action': 'searching',
                                    'red_detected': False,
                                    'pixel_count': 0
                                })
                                
                            except Exception as search_error:
                                print(f"❌ Autonomous search error: {search_error}")
                    
                time.sleep(0.1)
                    
            except Exception as e:
                print(f"❌ Autonomous behavior error: {e}")
                time.sleep(0.5)
        
        print("🤖 Autonomous behavior thread ended")


    def _web_integration_thread(self):
        """Handle web integration updates"""
        print("🌐 Web integration thread started")
        
        while self.running:
            try:
                # Update telemetry data if available
                if self.tello:
                    try:
                        # Get telemetry data (handle both string and dict)
                        state_data = self.tello.get_current_state()
                        telemetry = self._parse_telemetry(state_data)  # Pass raw data
                        
                        shared_data.update_status({
                            'telemetry': telemetry,
                            'flying': self.send_rc_control
                        })
                    except Exception as e:
                        print(f"⚠️ Telemetry update error: {e}")
                        pass
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"❌ Web integration error: {e}")
                time.sleep(1)
        
        print("🌐 Web integration thread ended")


    def _parse_telemetry(self, state_data):
        """Parse telemetry data with proper type conversion - Fixed version"""
        telemetry = {
            'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
            'speed_x': 0.0, 'speed_y': 0.0, 'speed_z': 0.0,
            'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0,
            'barometer': 0.0, 'tof': 0.0
        }
        
        try:
            if not state_data:
                return telemetry
                
            # Handle different return types from get_current_state()
            if isinstance(state_data, dict):
                # Handle dictionary format
                for key, value in state_data.items():
                    try:
                        value = float(value)  # Convert to Python float
                        
                        if key in ['pitch', 'roll', 'yaw']:
                            telemetry[key] = value
                        elif key in ['vgx', 'vgy', 'vgz']:
                            telemetry[f'speed_{key[-1]}'] = value
                        elif key in ['agx', 'agy', 'agz']:
                            telemetry[f'accel_{key[-1]}'] = value
                        elif key == 'baro':
                            telemetry['barometer'] = value
                        elif key == 'tof':
                            telemetry['tof'] = value
                    except (ValueError, TypeError):
                        pass  # Skip invalid values
                        
            elif isinstance(state_data, str):
                # Handle string format
                for item in state_data.split(';'):
                    if ':' in item:
                        key, value = item.split(':', 1)
                        try:
                            value = float(value)  # Convert to Python float
                            
                            if key in ['pitch', 'roll', 'yaw']:
                                telemetry[key] = value
                            elif key in ['vgx', 'vgy', 'vgz']:
                                telemetry[f'speed_{key[-1]}'] = value
                            elif key in ['agx', 'agy', 'agz']:
                                telemetry[f'accel_{key[-1]}'] = value
                            elif key == 'baro':
                                telemetry['barometer'] = value
                            elif key == 'tof':
                                telemetry['tof'] = value
                        except (ValueError, TypeError):
                            pass  # Skip invalid values
            else:
                print(f"⚠️ Unexpected telemetry data type: {type(state_data)}")
                
        except Exception as e:
            print(f"❌ Telemetry parsing error: {e}")
        
        return telemetry
    def _process_human_detection(self, frame):
        """Process human detection using YOLO"""
        try:
            output_frame = frame.copy()
            
            # YOLO Human Detection
            results = self.yolo_model(frame, verbose=False)
            
            detected = False
            human_count = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Check if it's a person (class_id = 0 in COCO dataset)
                        if class_id == 0 and confidence > 0.5:
                            detected = True
                            human_count += 1
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Draw bounding box
                            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Add label
                            label = f"Human: {confidence*100:.0f}%"
                            cv2.putText(output_frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return output_frame, detected, human_count
            
        except Exception as e:
            print(f"❌ Human detection error: {e}")
            return frame, False, 0
    
    def _execute_web_command(self, command):
        """Execute command from web interface"""
        try:
            cmd_type = command.get('type')
            cmd_data = command.get('data', {})
            
            print(f"🎯 Executing web command: {cmd_type}")
            
            if cmd_type == 'takeoff':
                if self.tello and not self.send_rc_control:
                    self.tello.takeoff()
                    self.send_rc_control = True
                    shared_data.update_status({'flying': True})
                    print("✅ Takeoff completed")
            
            elif cmd_type == 'land':
                if self.set_autonomous_behavior:
                    self.set_autonomous_behavior = False
                    self.send_rc_control = True
                    
                if self.tello and self.send_rc_control:
                    self.tello.land()
                    self.send_rc_control = False
                    shared_data.update_status({'flying': False})
                    print("✅ Landing completed")
            
            elif cmd_type == 'emergency':
                if self.tello:
                    self.tello.emergency()
                    self.send_rc_control = False
                    shared_data.update_status({'flying': False})
                    print("🚨 Emergency executed")
            
            elif cmd_type == 'enable_change_keyboard':
                enabled = cmd_data.get('enabled', False)
                shared_data.update_status({'keyboard_enabled': enabled})
                print(f"🎮 Keyboard mode updated: {'Mode 2 (Arrow Movement)' if enabled else 'Mode 1 (WASD Movement)'}")

            elif cmd_type == 'move_control':
                controls = cmd_data
                
                # Get current keyboard mode from shared data
                current_status = shared_data.get_status()
                keyboard_enabled = current_status.get('keyboard_enabled', False)
                
                # Process movement commands based on keyboard mode
                if keyboard_enabled:
                    # Mode 2: Frontend sends based on Arrow keys setup
                    # Frontend sudah mengirim nilai yang benar untuk Mode 2
                    self.left_right_velocity = controls.get('left_right', 0)
                    self.for_back_velocity = controls.get('for_back', 0)
                    self.up_down_velocity = controls.get('up_down', 0)
                    self.yaw_velocity = controls.get('yaw', 0)
                else:
                    # Mode 1: Frontend sends based on WASD setup  
                    # Frontend sudah mengirim nilai yang benar untuk Mode 1
                    self.left_right_velocity = controls.get('left_right', 0)
                    self.for_back_velocity = controls.get('for_back', 0)
                    self.up_down_velocity = controls.get('up_down', 0)
                    self.yaw_velocity = controls.get('yaw', 0)
                
                # Debug log untuk memastikan values diterima
                if any([self.left_right_velocity, self.for_back_velocity, self.up_down_velocity, self.yaw_velocity]):
                    mode_text = "Mode 2" if keyboard_enabled else "Mode 1"
                    print(f"🎮 Movement {mode_text}: LR={self.left_right_velocity}, FB={self.for_back_velocity}, UD={self.up_down_velocity}, YAW={self.yaw_velocity}")
                        
            elif cmd_type == 'stop_movement':
                self.left_right_velocity = 0
                self.for_back_velocity = 0
                self.up_down_velocity = 0
                self.yaw_velocity = 0
            
            elif cmd_type == 'set_speed':
                self.speed = max(10, min(100, cmd_data.get('speed', 50)))
                if self.tello:
                    self.tello.set_speed(self.speed)
                shared_data.update_status({'speed': self.speed})
                print(f"⚡ Speed set to: {self.speed}")
            
            elif cmd_type == 'flip':
                direction = cmd_data.get('direction', 'f')
                if self.tello and self.send_rc_control:
                    if direction == 'f':
                        self.tello.flip_forward()
                    elif direction == 'b':
                        self.tello.flip_back()
                    elif direction == 'l':
                        self.tello.flip_left()
                    elif direction == 'r':
                        self.tello.flip_right()
                    print(f"🔄 Flip {direction} executed")
            
            elif cmd_type == 'enable_ml_detection':
                self.detection_enabled = cmd_data.get('enabled', True)
                shared_data.update_status({'ml_detection_enabled': self.detection_enabled})
                print(f"🤖 ML Detection: {'ON' if self.detection_enabled else 'OFF'}")
 
            elif cmd_type == 'start_autonomous_mode':
                self.send_rc_control = False # Ensure RC control is enabled
                self.set_autonomous_behavior = True
                self.detection_enabled = True  # Ensure detection is enabled for autonomous mode
                shared_data.update_status({'autonomous_mode': True, 'ml_detection_enabled': True })
                
                print("🤖 Autonomous mode: STARTED")
                
                # Ensure drone is flying for autonomous mode
                if not self.send_rc_control:
                    print("🛫 Auto-takeoff for autonomous mode")
                    self.tello.takeoff()
                    shared_data.update_status({'flying': True})
            
            elif cmd_type == 'stop_autonomous_mode':
                self.set_autonomous_behavior = False
                shared_data.update_status({'autonomous_mode': False})
                print("🤖 Autonomous mode: STOPPED")
            
            elif cmd_type == 'manual_screenshot':
                self._take_screenshot()
            
            elif cmd_type == 'toggle_recording':
                self._toggle_recording()
            
        except Exception as e:
            print(f"❌ Command execution error: {e}")
    
    def _take_screenshot(self):
        """Take a screenshot"""
        try:
            if self.current_processed_frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"web_screenshot_{timestamp}.jpg"
                filepath = os.path.join(Config.SCREENSHOTS_DIR, filename)
                
                frame_bgr = cv2.cvtColor(self.current_processed_frame, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(filepath, frame_bgr)
                
                if success:
                    current_count = shared_data.get_status().get('screenshot_count', 0) + 1
                    shared_data.update_status({'screenshot_count': current_count})
                    print(f"📸 Screenshot saved: {filename}")
                    return True
        except Exception as e:
            print(f"❌ Screenshot error: {e}")
        return False
    
    def _toggle_recording(self):
        """Toggle video recording"""
        try:
            if not self.recording:
                # Start recording
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"web_recording_{timestamp}.mp4"
                filepath = os.path.join(Config.RECORDINGS_DIR, filename)
                
                fourcc = cv2.VideoWriter_fourcc(*'h264')
                self.video_writer = cv2.VideoWriter(filepath, fourcc, 30.0, 
                                                   (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
                
                if self.video_writer.isOpened():
                    self.recording = True
                    shared_data.update_status({'recording': True})
                    print(f"🔴 Recording started: {filename}")
                    return True
            else:
                # Stop recording
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
                
                self.recording = False
                shared_data.update_status({'recording': False})
                print("⏹️ Recording stopped")
                return True
        except Exception as e:
            print(f"❌ Recording error: {e}")
        return False
    
    def stop_all_systems(self):
        """Stop all drone systems"""
        print("🛑 Stopping drone systems...")
        self.running = False
        
        # Stop recording if active
        if self.recording:
            self._toggle_recording()
        
        # Cleanup Tello
        if self.tello:
            try:
                self.tello.streamoff()
                self.tello.end()
            except:
                pass
        
        # Cleanup AI models
        if self.pose:
            self.pose.close()
        if self.hands:
            self.hands.close()
        
        pygame.quit()
        print("✅ Drone systems stopped")

# ==================== WEB SERVER ====================
class WebServer:
    """Flask + Socket.IO web server for React frontend"""
    
    def __init__(self):
        if not WEB_IMPORTS_AVAILABLE:
            raise ImportError("Web server dependencies not available")
        
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'drone_web_bridge_secret'
        
        # Enable CORS
        CORS(self.app, 
            origins="*",
            methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            allow_headers=['Content-Type', 'Authorization', 'Access-Control-Allow-Credentials'],
            supports_credentials=True)

        # Initialize Socket.IO
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='threading',
            logger=False,
            engineio_logger=False
        )
        
        self.setup_routes()
        self.setup_socket_events()
        
        print("🌐 Web server initialized")
    
    def setup_routes(self):
        """Setup REST API routes"""
        
        @self.app.route('/')
        def index():
            return jsonify({
                'message': 'Drone Web Bridge Server',
                'status': 'running',
                'endpoints': {
                    'status': '/api/status',
                    'socket': '/socket.io'
                }
            })

        @self.app.route('/api/media/list')
        def list_media_files():
            """List media files with pagination and filtering"""
            try:
                file_type = request.args.get('type', 'images').lower()
                
                if file_type == 'images':
                    directory = Config.SCREENSHOTS_DIR
                    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                elif file_type == 'videos':
                    directory = Config.RECORDINGS_DIR  
                    extensions = ['.mp4', '.avi', '.mov', '.mkv']
                else:
                    return jsonify({'success': False, 'error': 'Invalid file type'})
                
                files = []
                if os.path.exists(directory):
                    for filename in os.listdir(directory):
                        if any(filename.lower().endswith(ext) for ext in extensions):
                            filepath = os.path.join(directory, filename)
                            stat = os.stat(filepath)
                            
                            files.append({
                                'filename': filename,
                                'size': stat.st_size,
                                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                'url': f"{Config.MEDIA_BASE_URL}/{filename}",
                                'file_path': filepath,
                                'type': file_type[:-1]  # Remove 's' from images/videos
                            })
                
                # Sort by creation time (newest first)
                files.sort(key=lambda x: x['created_at'], reverse=True)
                
                return jsonify({
                    'success': True,
                    'files': files,
                    'count': len(files),
                    'type': file_type
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/media/<filename>')
        def serve_media_file(filename):
            """Serve media files from screenshots or recordings directory"""
            try:
                # Try screenshots directory first
                screenshots_path = os.path.join(Config.SCREENSHOTS_DIR, filename)
                if os.path.exists(screenshots_path):
                    return send_file(screenshots_path, as_attachment=False)
                
                # Try recordings directory
                recordings_path = os.path.join(Config.RECORDINGS_DIR, filename) 
                if os.path.exists(recordings_path):
                    return send_file(recordings_path, as_attachment=False)
                
                return jsonify({'error': 'File not found'}), 404
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/download/<filename>')
        def download_media_file(filename):
            """Download media files"""
            try:
                # Try screenshots directory first
                screenshots_path = os.path.join(Config.SCREENSHOTS_DIR, filename)
                if os.path.exists(screenshots_path):
                    return send_file(screenshots_path, as_attachment=True, download_name=filename)
                
                # Try recordings directory
                recordings_path = os.path.join(Config.RECORDINGS_DIR, filename)
                if os.path.exists(recordings_path):
                    return send_file(recordings_path, as_attachment=True, download_name=filename)
                
                return jsonify({'error': 'File not found'}), 404
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/media/delete/<filename>', methods=['DELETE'])
        def delete_media_file(filename):
            """Delete media file"""
            try:
                # Try screenshots directory first  
                screenshots_path = os.path.join(Config.SCREENSHOTS_DIR, filename)
                if os.path.exists(screenshots_path):
                    os.remove(screenshots_path)
                    return jsonify({'success': True, 'message': f'File {filename} deleted'})
                
                # Try recordings directory
                recordings_path = os.path.join(Config.RECORDINGS_DIR, filename)
                if os.path.exists(recordings_path):
                    os.remove(recordings_path)
                    return jsonify({'success': True, 'message': f'File {filename} deleted'})
                
                return jsonify({'success': False, 'error': 'File not found'}), 404
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/status')
        def get_status():
            return jsonify({
                'success': True,
                'data': shared_data.get_status()
            })

    def setup_socket_events(self):
        """Setup Socket.IO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"🔗 Client connected: {request.sid}")
            # Send current status
            emit('tello_status', shared_data.get_status())
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"🔌 Client disconnected: {request.sid}")
        
        @self.socketio.on('connect_tello')
        def handle_connect_tello():
            """Handle Tello connection request"""
            # Tello should already be connected in integrated mode
            status = shared_data.get_status()
            emit('tello_status', status)
        
        @self.socketio.on('disconnect_tello')
        def handle_disconnect_tello():
            """Handle Tello disconnection request"""
            shared_data.add_command({'type': 'emergency'})
            emit('tello_status', {'connected': False, 'flying': False})

        @self.socketio.on('get_media_files')
        def handle_get_media_files(data):
            """Handle request for media files list"""
            try:
                file_type = data.get('type', 'images').lower()
                
                if file_type == 'images':
                    directory = Config.SCREENSHOTS_DIR
                    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                elif file_type == 'videos':
                    directory = Config.RECORDINGS_DIR
                    extensions = ['.mp4', '.avi', '.mov', '.mkv']
                else:
                    emit('media_files_response', {'success': False, 'error': 'Invalid file type'})
                    return
                
                files = []
                if os.path.exists(directory):
                    for filename in os.listdir(directory):
                        if any(filename.lower().endswith(ext) for ext in extensions):
                            filepath = os.path.join(directory, filename)
                            stat = os.stat(filepath)
                            
                            files.append({
                                'filename': filename,
                                'size': stat.st_size,
                                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                'url': f"{Config.MEDIA_BASE_URL}/{filename}",
                                'file_path': filepath,
                                'type': file_type[:-1]
                            })
                
                # Sort by creation time (newest first)
                files.sort(key=lambda x: x['created_at'], reverse=True)
                
                emit('media_files_response', {
                    'success': True,
                    'files': files,
                    'count': len(files),
                    'type': file_type
                })
                
            except Exception as e:
                emit('media_files_response', {'success': False, 'error': str(e)})

        @self.socketio.on('delete_media')
        def handle_delete_media(data):
            """Handle media file deletion"""
            try:
                filename = data.get('filename')
                if not filename:
                    emit('media_deleted', {'success': False, 'error': 'Filename required'})
                    return
                
                # Try screenshots directory first
                screenshots_path = os.path.join(Config.SCREENSHOTS_DIR, filename)
                if os.path.exists(screenshots_path):
                    os.remove(screenshots_path)
                    emit('media_deleted', {'success': True, 'filename': filename})
                    return
                
                # Try recordings directory
                recordings_path = os.path.join(Config.RECORDINGS_DIR, filename)
                if os.path.exists(recordings_path):
                    os.remove(recordings_path)
                    emit('media_deleted', {'success': True, 'filename': filename})
                    return
                
                emit('media_deleted', {'success': False, 'error': 'File not found'})
                
            except Exception as e:
                emit('media_deleted', {'success': False, 'error': str(e)})

        @self.socketio.on('download_media')
        def handle_download_media(data):
            """Handle media file download preparation"""
            try:
                filename = data.get('filename')
                if not filename:
                    emit('download_ready', {'success': False, 'error': 'Filename required'})
                    return
                
                # Check if file exists and prepare download URL
                screenshots_path = os.path.join(Config.SCREENSHOTS_DIR, filename)
                recordings_path = os.path.join(Config.RECORDINGS_DIR, filename)
                
                if os.path.exists(screenshots_path) or os.path.exists(recordings_path):
                    download_url = f"http://{Config.WEB_HOST}:{Config.WEB_PORT}/download/{filename}"
                    emit('download_ready', {
                        'success': True,
                        'url': download_url,
                        'filename': filename
                    })
                else:
                    emit('download_ready', {'success': False, 'error': 'File not found'})
                    
            except Exception as e:
                emit('download_ready', {'success': False, 'error': str(e)})

        @self.socketio.on('debug_media_system')
        def handle_debug_media():
            """Handle debug request for media system"""
            try:
                debug_info = {
                    'directories': {
                        'screenshots': Config.SCREENSHOTS_DIR,
                        'recordings': Config.RECORDINGS_DIR,
                        'screenshots_exists': os.path.exists(Config.SCREENSHOTS_DIR),
                        'recordings_exists': os.path.exists(Config.RECORDINGS_DIR)
                    },
                    'file_counts': {
                        'screenshots': len([f for f in os.listdir(Config.SCREENSHOTS_DIR) 
                                        if os.path.isfile(os.path.join(Config.SCREENSHOTS_DIR, f))]) 
                                    if os.path.exists(Config.SCREENSHOTS_DIR) else 0,
                        'recordings': len([f for f in os.listdir(Config.RECORDINGS_DIR) 
                                        if os.path.isfile(os.path.join(Config.RECORDINGS_DIR, f))]) 
                                    if os.path.exists(Config.RECORDINGS_DIR) else 0
                    },
                    'permissions': {
                        'screenshots_readable': os.access(Config.SCREENSHOTS_DIR, os.R_OK) if os.path.exists(Config.SCREENSHOTS_DIR) else False,
                        'recordings_readable': os.access(Config.RECORDINGS_DIR, os.R_OK) if os.path.exists(Config.RECORDINGS_DIR) else False
                    }
                }
                
                emit('debug_media_response', {'debug_info': debug_info})
                
            except Exception as e:
                emit('debug_media_response', {'debug_info': {'error': str(e)}})

        @self.socketio.on('takeoff')
        def handle_takeoff():
            """Handle takeoff command"""
            shared_data.add_command({'type': 'takeoff'})
            emit('drone_action', {'action': 'takeoff', 'success': True})
        
        @self.socketio.on('land')
        def handle_land():
            """Handle land command"""
            shared_data.add_command({'type': 'land'})
            emit('drone_action', {'action': 'land', 'success': True})

        @self.socketio.on('enable_change_keyboard')
        def handle_enable_change_keyboard(data):
            """Handle keyboard mode change"""
            enabled = data.get('enabled', False)
            
            # Update shared data status
            shared_data.update_status({'keyboard_enabled': enabled})
            
            # Add command to drone system
            shared_data.add_command({
                'type': 'enable_change_keyboard',
                'data': {'enabled': enabled}
            })
            
            # Send confirmation back to frontend
            emit('keyboard_mode_updated', {
                'enabled': enabled,
                'mode': 'Mode 2: Arrow Movement' if enabled else 'Mode 1: WASD Movement'
            })
            
            print(f"🎮 Keyboard mode changed to: {'Mode 2 (Arrow Movement)' if enabled else 'Mode 1 (WASD Movement)'}")
                
        @self.socketio.on('emergency_land')
        def handle_emergency():
            """Handle emergency command"""
            shared_data.add_command({'type': 'emergency'})
            emit('drone_action', {'action': 'emergency', 'success': True})
        
        @self.socketio.on('move_control')
        def handle_move_control(data):
            """Handle movement control"""
            shared_data.add_command({
                'type': 'move_control',
                'data': {
                    'left_right': data.get('left_right', 0),
                    'for_back': data.get('for_back', 0),
                    'up_down': data.get('up_down', 0),
                    'yaw': data.get('yaw', 0)
                }
            })
        
        @self.socketio.on('stop_movement')
        def handle_stop_movement():
            """Handle stop movement command"""
            shared_data.add_command({'type': 'stop_movement'})
        
        @self.socketio.on('set_speed')
        def handle_set_speed(data):
            """Handle speed setting"""
            speed = data.get('speed', 20)
            shared_data.add_command({
                'type': 'set_speed',
                'data': {'speed': speed}
            })
            emit('speed_update', {'speed': speed})
        
        @self.socketio.on('flip_command')
        def handle_flip(data):
            """Handle flip command"""
            direction = data.get('direction', 'f')
            shared_data.add_command({
                'type': 'flip',
                'data': {'direction': direction}
            })
            emit('drone_action', {'action': f'flip_{direction}', 'success': True})
        
        @self.socketio.on('enable_ml_detection')
        def handle_enable_ml_detection(data):
            """Handle ML detection toggle"""
            enabled = data.get('enabled', True)
            shared_data.add_command({
                'type': 'enable_ml_detection',
                'data': {'enabled': enabled}
            })
            emit('ml_detection_status', {'enabled': enabled})

        @self.socketio.on('enable_auto_capture')
        def handle_enable_auto_capture(data):
            """Handle auto capture toggle"""
            enabled = data.get('enabled', True)
            shared_data.update_status({'auto_capture_enabled': enabled})
            emit('auto_capture_status', {'enabled': enabled})
        
        @self.socketio.on('manual_screenshot')
        def handle_manual_screenshot():
            """Handle manual screenshot request"""
            shared_data.add_command({'type': 'manual_screenshot'})
            
            # Simulate screenshot result
            current_count = shared_data.get_status().get('screenshot_count', 0)
            emit('screenshot_result', {
                'success': True,
                'count': current_count + 1,
                'filename': f'screenshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
            })
        
        @self.socketio.on('toggle_recording')
        def handle_toggle_recording(data):
            """Handle recording toggle"""
            recording = data.get('recording', False)
            shared_data.add_command({'type': 'toggle_recording'})
            emit('recording_status', {'recording': recording})
        
        @self.socketio.on('start_stream')
        def handle_start_stream():
            """Handle video stream start"""
            emit('stream_status', {'streaming': True})
            # Start sending video frames
            self.start_video_stream()
        
        @self.socketio.on('start_autonomous_mode')
        def handle_start_autonomous():
            """Handle autonomous mode start"""
            shared_data.add_command({'type': 'start_autonomous_mode'})
            emit('drone_action', {'action': 'autonomous_start', 'success': True})
            emit('autonomous_status', {'enabled': True})
            print("🤖 Autonomous mode start command sent")
        
        @self.socketio.on('stop_autonomous_mode')
        def handle_stop_autonomous():
            """Handle autonomous mode stop"""
            shared_data.add_command({'type': 'stop_autonomous_mode'})
            emit('drone_action', {'action': 'autonomous_stop', 'success': True})
            emit('autonomous_status', {'enabled': False})
            print("🤖 Autonomous mode stop command sent")


    def start_video_stream(self):
        """Start video streaming to clients"""
        def video_stream_worker():
            while True:
                try:
                    frame_base64 = shared_data.get_frame_base64()
                    if frame_base64:
                        self.socketio.emit('video_frame', {
                            'frame': frame_base64,
                            'timestamp': time.time()
                        })
                    
                    time.sleep(1/30)  # 30 FPS
                except Exception as e:
                    print(f"❌ Video stream error: {e}")
                    time.sleep(0.1)
        
        # Start video streaming in separate thread
        video_thread = threading.Thread(target=video_stream_worker, daemon=True)
        video_thread.start()
    
    def start_status_updates(self):
        """Start periodic status updates"""
        def status_update_worker():
            while True:
                try:
                    status = shared_data.get_status()
                    self.socketio.emit('tello_status', status)
                    self.socketio.emit('telemetry_update', status.get('telemetry', {}))
                    
                    # Send autonomous status updates
                    if status.get('autonomous_mode', False):
                        self.socketio.emit('autonomous_update', {
                            'enabled': status.get('autonomous_mode', False),
                            'action': status.get('autonomous_action', 'idle'),
                            'red_detected': status.get('red_detected', False),
                            'pixel_count': status.get('pixel_count', 0)
                        })
                    
                    time.sleep(2)  # Update every 2 seconds
                except Exception as e:
                    print(f"❌ Status update error: {e}")
                    time.sleep(1)
        
        # Start status updates in separate thread
        status_thread = threading.Thread(target=status_update_worker, daemon=True)
        status_thread.start()
    
    def run(self):
        """Run the web server"""
        print(f"🚀 Starting web server on {Config.WEB_HOST}:{Config.WEB_PORT}")
        
        # Start video streaming and status updates
        self.start_video_stream()
        self.start_status_updates()
        
        # Run the server
        self.socketio.run(
            self.app,
            host=Config.WEB_HOST,
            port=Config.WEB_PORT,
            debug=Config.WEB_DEBUG,
            allow_unsafe_werkzeug=True
        )

# ==================== MAIN APPLICATION ====================
class DroneWebBridge:
    """Main application class that orchestrates everything"""
    
    def __init__(self, mode='integrated'):
        self.mode = mode
        self.drone_system = None
        self.web_server = None
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(f"🚀 Drone Web Bridge starting in {mode} mode")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def start_backend_only(self):
        """Start only web server (for testing without drone)"""
        if not WEB_IMPORTS_AVAILABLE:
            print("❌ Web server dependencies not available")
            return False
        
        try:
            # Initialize with mock data
            shared_data.update_status({
                'connected': False,
                'flying': False,
                'battery': 0,
                'speed': 20
            })
            
            # Start web server
            self.web_server = WebServer()
            print("🌐 Backend-only mode: Web server ready")
            print(f"📍 Access at: http://{Config.WEB_HOST}:{Config.WEB_PORT}")
            print("⚠️ Drone system not initialized - limited functionality")
            
            self.web_server.run()
            
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
    
    def start_drone_only(self):
        """Start only drone system (original droneV7.py behavior)"""
        if not DRONE_IMPORTS_AVAILABLE:
            print("❌ Drone dependencies not available")
            return False
        
        try:
            # Initialize drone system in standalone mode
            self.drone_system = DroneSystem()
            self.drone_system.web_integration_enabled = False
            
            if not self.drone_system.initialize_all_systems():
                print("❌ Failed to initialize drone systems")
                return False
            
            # Start drone threads
            self.drone_system.start_drone_threads()
            
            print("🚁 Drone-only mode: Running original droneV7.py behavior")
            print("🎮 Use pygame interface for control")
            
            # Run original main loop (simplified)
            try:
                while self.running:
                    # Handle pygame events if not in headless mode
                    if self.drone_system.screen:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                self.running = False
                                break
                    
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n🛑 Keyboard interrupt received")
            
        except Exception as e:
            print(f"❌ Failed to start drone system: {e}")
            return False
    
    def start_integrated(self):
        if not WEB_IMPORTS_AVAILABLE or not DRONE_IMPORTS_AVAILABLE:
            print("❌ Missing dependencies for integrated mode")
            return False
        
        try:
            self.drone_system = DroneSystem()
            self.drone_system.web_integration_enabled = True
            
            if not self.drone_system.initialize_all_systems():
                print("❌ Failed to initialize drone systems")
                return False
            
            # Start drone threads
            self.drone_system.start_drone_threads()
            self.web_server = WebServer()
            
            # Run web server (this will block)
            self.web_server.run()
            
        except Exception as e:
            print(f"❌ Failed to start integrated mode: {e}")
            return False
    
def main():
    """Main entry point"""
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--help', '-h']:
            return
        elif arg == '--backend-only':
            mode = 'backend-only'
        elif arg == '--drone-only':
            mode = 'drone-only'
        elif arg == '--integrated':
            mode = 'integrated'
        else:
            print(f"❌ Unknown mode: {arg}")
            return
    else:
        mode = 'integrated'  # Default mode
    
    # Create and start application
    app = DroneWebBridge(mode)
    
    try:
        if mode == 'backend-only':
            success = app.start_backend_only()
        elif mode == 'drone-only':
            success = app.start_drone_only()
        elif mode == 'integrated':
            success = app.start_integrated()
        
        if not success:
            print("❌ Failed to start application")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

'''