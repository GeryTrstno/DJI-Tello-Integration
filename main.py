from djitellopy import Tello
import cv2, queue, threading, time, pygame, os, sys
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import deque

# Speed and FPS settings
FPS = 120

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

# ROI parameters (Region of Interest)
ROI_X = WINDOW_WIDTH // 4
ROI_Y = 5
ROI_WIDTH = WINDOW_WIDTH // 2
ROI_HEIGHT = WINDOW_HEIGHT // 3

# Thread control
running = True
threads = []

# Thread-safe queues
frame_queue = queue.Queue(maxsize=5)
command_queue = queue.Queue()
screenshot_queue = queue.Queue()

# Shared data with locks
data_lock = threading.Lock()
current_frame = None
current_processed_frame = None
current_detection = None  # FIX: Initialize current_detection
battery_level = 0
fps = 0
humans_count = 0
human_detected = False
screenshot_count = 0

# Performance monitoring
frame_times = deque(maxlen=30)
last_frame_time = time.time()

# Drone control variables
for_back_velocity = 0
left_right_velocity = 0
up_down_velocity = 0
yaw_velocity = 0
speed = 10
send_rc_control = False

# Screenshot variables
last_screenshot_time = 0
screenshot_interval = 3
countdown_active = False
countdown_start_time = 0
countdown_duration = 3.0
last_human_detected = False

set_autonomous_behavior = False

# Joystick screenshot variables
last_joystick_screenshot_button_state = False
joystick_screenshot_requested = False

# Global objects
tello = None
screen = None
joystick = None
screenshot_dir = "screenshots"
yolo_model = None
mp_drawing = None
mp_drawing_styles = None
mp_pose = None
mp_hands = None
pose = None
hands = None

def initialize_pygame():
    """Initialize pygame and create window"""
    global screen, joystick
    
    pygame.init()
    pygame.display.set_caption("Tello video stream")
    screen = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])
    
    # Initialize joystick
    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Joystick initialized: {joystick.get_name()}")
    else:
        print("No joystick detected - using keyboard only")
    
    # Create update timer
    pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

def initialize_tello():
    """Initialize Tello drone connection"""
    global tello, battery_level
    
    try:
        tello = Tello()
        tello.connect()
        tello.set_speed(speed)
        battery_level = tello.get_battery()  # FIX: Initialize battery level
        print(f"Battery: {battery_level}%")
        return True
    except Exception as e:
        print(f"Failed to connect to Tello: {e}")
        return False

def initialize_ai_models():
    """Initialize AI models for detection"""
    global yolo_model, mp_drawing, mp_drawing_styles, mp_pose, mp_hands, pose, hands
    
    try:
        # Load YOLOv8 model for human detection
        print("Loading YOLOv8 model...")
        yolo_model = YOLO('yolov8n.pt')
        
        # Mediapipe modules for body part detection
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        
        # Create MediaPipe models
        pose = mp_pose.Pose(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True
        )
        
        hands = mp_hands.Hands(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            max_num_hands=2
        )
        
        # FIX: Initialize current_detection with default values
        global current_detection
        current_detection = {
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

def create_screenshot_directory():
    """Create screenshots directory if it doesn't exist"""
    try:
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)
            print(f"Created directory: {screenshot_dir}")
    except Exception as e:
        print(f"Failed to create screenshot directory: {e}")

def process_human_detection(frame):
    """Process human detection and return processed frame with detection info"""
    try:
        # Make a copy for processing
        output_frame = frame.copy()
        
        # YOLOv8 Human Detection
        results = yolo_model(frame, verbose=False)

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
                    if class_id == 0 and confidence > 0.5:
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
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with pose detection
            pose_results = pose.process(rgb_frame)

            # Process with hand detection
            hands_results = hands.process(rgb_frame)

            # Draw hands if detected
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        output_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Draw pose landmarks if detected
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    output_frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

        return output_frame, detected, len(human_boxes)
    
    except Exception as e:
        print(f"Error in human detection: {e}")
        return frame, False, 0

def handle_auto_screenshot(output_frame, human_detected_now, humans_count_now):
    """Handle auto screenshot countdown logic"""
    global countdown_active, countdown_start_time, last_human_detected
    
    try:
        current_time = time.time()
        
        if human_detected_now and humans_count_now >= 1:
            # Human detected
            if not last_human_detected and not countdown_active:
                # First time detecting human, start countdown
                countdown_active = True
                countdown_start_time = current_time
                print(f"Human detected! Starting 3-second countdown...")
            
            # If countdown is active
            if countdown_active:
                elapsed_time = current_time - countdown_start_time
                
                if elapsed_time >= countdown_duration:
                    # Countdown finished, take screenshot
                    save_screenshot(output_frame.copy(), humans_count_now, "auto")
                    countdown_active = False
                    print("Countdown completed! Screenshot taken.")
        else:
            # No human detected
            if countdown_active:
                # Cancel countdown if human disappears
                countdown_active = False
                print("Human detection lost! Countdown cancelled.")
        
        # Update human detection status for next frame
        last_human_detected = human_detected_now
        
    except Exception as e:
        print(f"Error in auto screenshot handler: {e}")

def save_screenshot(frame, humans_count_param, source="auto"):
    """Save screenshot with timestamp and human count"""
    global screenshot_count
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_prefix = "manual" if source in ["joystick", "keyboard"] else "auto"
        filename = f"{source_prefix}_human_detected_{timestamp}_{humans_count_param}persons_{screenshot_count:04d}.jpg"
        filepath = os.path.join(screenshot_dir, filename)
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for saving

        # Save the frame
        success = cv2.imwrite(filepath, frame_bgr)
        
        if success:
            screenshot_count += 1
            print(f"Screenshot saved ({source}): {filename}")
            return True
        else:
            print(f"Failed to save screenshot: {filename}")
            return False
    
    except Exception as e:
        print(f"Error saving screenshot: {e}")
        return False

def video_stream_thread():
    """Thread 1: Handle video capture, processing, and screenshot operations"""
    global running, current_frame, current_processed_frame, human_detected, humans_count, fps, frame_times
    
    print("Video stream thread started")
    
    try:
        frame_read = tello.get_frame_read()
        
        while running:
            try:
                if frame_read.stopped:
                    break
                    
                frame = frame_read.frame
                if frame is not None:
                    # Resize frame once here
                    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
                    
                    # Process human detection immediately
                    output_frame, detected, count = process_human_detection(frame)
                    
                    # Handle auto screenshot logic
                    handle_auto_screenshot(output_frame, detected, count)
                    
                    # Process any screenshot requests from queue
                    while not screenshot_queue.empty():
                        try:
                            screenshot_data = screenshot_queue.get_nowait()
                            frame_to_save, humans_count_param, source = screenshot_data
                            save_screenshot(frame_to_save, humans_count_param, source)
                        except queue.Empty:
                            break
                    
                    # Update shared data
                    with data_lock:
                        current_frame = frame.copy()
                        current_processed_frame = output_frame.copy()
                        human_detected = detected
                        humans_count = count
                        
                        # Calculate FPS
                        current_time = time.time()
                        frame_times.append(current_time)
                        if len(frame_times) > 1:
                            time_diff = frame_times[-1] - frame_times[0]
                            fps = len(frame_times) / time_diff if time_diff > 0 else 0
                        
                time.sleep(1/60)  # Limit to 60 FPS capture
                
            except Exception as e:
                print(f"Video stream error: {e}")
                time.sleep(0.1)  # Brief pause before retrying
        
    except Exception as e:
        print(f"Critical video stream error: {e}")
    
    print("Video stream thread ended")

def execute_drone_command(command):
    """Execute drone commands safely"""
    global send_rc_control
    
    try:
        if command == "takeoff":
            tello.takeoff()
            send_rc_control = True
        elif command == "land":
            tello.land()
            send_rc_control = False
        elif command == "emergency":
            tello.emergency()
            send_rc_control = False
    except Exception as e:
        print(f"Command execution error: {e}")

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
        lower_red1 = np.array([0, 120, 130])
        upper_red1 = np.array([30, 255, 255])
        lower_red2 = np.array([230, 150, 130])
        upper_red2 = np.array([255, 255, 255])


        mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        mask_roi = cv2.bitwise_or(mask1, mask2)
        
        # Check if red color is detected
        red_detected = np.sum(mask_roi) > 10000 
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
        lower_red1 = np.array([0, 120, 130])
        upper_red1 = np.array([30, 255, 255])
        lower_red2 = np.array([230, 150, 130])
        upper_red2 = np.array([255, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        result = cv2.bitwise_and(img, img, mask=mask)
        return mask, result
    
    except Exception as e:
        print(f"Error in red color detection: {e}")
        return None, None

def detection_thread():
    """Thread for red color detection"""
    global current_detection
    
    print("Detection thread started")
    
    while running:
        try:
            # Get current frame safely
            with data_lock:
                if current_processed_frame is not None:
                    frame_copy = current_processed_frame.copy()
                else:
                    frame_copy = None
            
            if frame_copy is not None:

                frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                # Perform detection
                mask, result = detect_red_color(frame_copy)
                red_in_roi, roi_mask, full_roi_mask, pixel_count = detect_red_in_roi(frame_copy)
                
                # Update detection results safely
                with data_lock:
                    current_detection = {
                        'red_detected': red_in_roi,
                        'mask': mask,
                        'result': result,
                        'roi_mask': roi_mask,
                        'full_roi_mask': full_roi_mask,
                        'pixel_count': pixel_count
                    }
            
            time.sleep(0.03)  # ~30 FPS detection rate
            
        except Exception as e:
            print(f"Detection thread error: {e}")
            time.sleep(0.1)
    
    print("Detection thread ended")

def autonomous_behavior_thread():
    """Thread for autonomous behavior based on detection results"""
    print("Autonomous behavior thread started")
    
    while running:
        try:
            if set_autonomous_behavior and current_detection is not None:
                # Get current detection results safely
                with data_lock:
                    if current_detection:  # FIX: Check if current_detection exists
                        red_detected = current_detection.get('red_detected', False)
                        pixel_count = current_detection.get('pixel_count', 0)
                    else:
                        red_detected = False
                        pixel_count = 0
                
                # Autonomous behavior logic
                if red_detected:
                    print("🔴 Red detected in ROI! Moving towards target...")
                    tello.move_back(70)
                    time.sleep(2)
                    tello.rotate_clockwise(90)
                    time.sleep(2)
                else:
                    print("⚪ No red in ROI. Searching...")
                    tello.move_forward(30)
                    time.sleep(1)
            else:
                time.sleep(0.1)  # Sleep when autonomous mode is disabled
                
        except Exception as e:
            print(f"Autonomous behavior error: {e}")
            time.sleep(0.5)
    
    print("Autonomous behavior thread ended")

def drone_control_thread():
    """Thread 2: Handle drone control commands and battery monitoring"""
    global running, battery_level
    
    print("Drone control thread started")
    last_battery_check = time.time()
    
    while running:
        try:
            # Process any queued commands
            while not command_queue.empty():
                try:
                    command = command_queue.get_nowait()
                    execute_drone_command(command)
                except queue.Empty:
                    break
            
            # Send RC control if active
            if send_rc_control:
                tello.send_rc_control(
                    left_right_velocity, 
                    for_back_velocity,
                    up_down_velocity, 
                    yaw_velocity
                )
            
            # Update battery level periodically (every 10 seconds)
            current_time = time.time()
            if current_time - last_battery_check >= 10:
                try:
                    with data_lock:
                        battery_level = tello.get_battery()
                    last_battery_check = current_time
                except Exception as e:
                    print(f"Battery check error: {e}")
            
            time.sleep(1/30)  # 30 Hz control rate
            
        except Exception as e:
            print(f"Drone control error: {e}")
            time.sleep(0.1)
    
    print("Drone control thread ended")

def get_joystick_input():
    """Get joystick input"""
    global left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity
    global last_joystick_screenshot_button_state, send_rc_control
    
    if not joystick:
        return

    try:
        speed_joy = 50
        rotate = 80

        # Read joystick input
        axis_lr = joystick.get_axis(0)  # Left-right movement
        axis_fb = joystick.get_axis(1)  # Forward-backward movement
        axis_yv = joystick.get_axis(2)  # Up-down movement  
        axis_ud = joystick.get_axis(3)  # Yaw rotation

        # Set velocities based on joystick input
        left_right_velocity = int(axis_lr * speed_joy)
        for_back_velocity = int(-axis_fb * speed_joy)
        up_down_velocity = int(-axis_ud * speed_joy)
        yaw_velocity = int(axis_yv * rotate)

        # Handle buttons
        if joystick.get_button(0):  # Button A - takeoff
            if not send_rc_control:
                command_queue.put("takeoff")
                time.sleep(0.5)

        if joystick.get_button(1):  # Button B - land
            if send_rc_control:
                command_queue.put("land")
                time.sleep(0.5)

        # Screenshot buttons
        current_screenshot_button_state = joystick.get_button(2)
        if current_screenshot_button_state and not last_joystick_screenshot_button_state:
            request_manual_screenshot("joystick")
        
        last_joystick_screenshot_button_state = current_screenshot_button_state

        if joystick.get_button(3):  # Alternative screenshot button
            request_manual_screenshot("joystick")
            time.sleep(0.2)
        
        if joystick.get_button(6):
            global set_autonomous_behavior
            set_autonomous_behavior = True
            print("Autonomous behavior enabled")

        elif joystick.get_button(7):
            set_autonomous_behavior = False
            print("Autonomous behavior disabled")
        
        if joystick.get_button(8):  # Left shoulder - enable RC control
            send_rc_control = True
            print("🎮 RC Control: ENABLED")

        if joystick.get_button(9):  # Right shoulder - disable RC control
            send_rc_control = False
            left_right_velocity = 0
            for_back_velocity = 0
            up_down_velocity = 0
            yaw_velocity = 0
            print("🎮 RC Control: DISABLED")

    except Exception as e:
        print(f"Joystick input error: {e}")

def request_manual_screenshot(source):
    """Request a manual screenshot"""
    try:
        with data_lock:
            if current_processed_frame is not None:
                screenshot_queue.put((
                    current_processed_frame.copy(), 
                    humans_count, 
                    source
                ))
                print(f"Manual screenshot requested ({source})")
    except Exception as e:
        print(f"Screenshot request error: {e}")

def draw_roi_rectangle(surface):
    """Draw ROI rectangle on pygame surface"""
    try:
        roi_color = (0, 255, 0)  # Green color for ROI
        roi_rect = pygame.Rect(ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT)
        pygame.draw.rect(surface, roi_color, roi_rect, 3)  # 3 pixel thick border
    except Exception as e:
        print(f"ROI drawing error: {e}")

def start_threads():
    """Start worker threads"""
    global threads
    
    # Start video stream thread
    video_thread = threading.Thread(target=video_stream_thread, daemon=True)
    video_thread.start()
    threads.append(video_thread)
    
    # Start drone control thread
    control_thread = threading.Thread(target=drone_control_thread, daemon=True)
    control_thread.start()
    threads.append(control_thread)

    # Start detection thread
    detection_thread_instance = threading.Thread(target=detection_thread, daemon=True)
    detection_thread_instance.start()
    threads.append(detection_thread_instance)

    # Start autonomous behavior thread
    autonomous_thread = threading.Thread(target=autonomous_behavior_thread, daemon=True)
    autonomous_thread.start()
    threads.append(autonomous_thread)
    
    print(f"Started {len(threads)} worker threads")

def stop_threads():
    """Stop all worker threads"""
    global running
    
    print("Stopping threads...")
    running = False
    
    # Wait for threads to finish
    for thread in threads:
        if thread.is_alive():
            thread.join(timeout=2)
    
    print("All threads stopped")

def main_loop():
    """Main run loop - handles UI and input"""
    global running
    
    should_stop = False
    
    try:
        while not should_stop:

            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    pass  # Update timer
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    elif event.key == pygame.K_p:  # FIX: Add keyboard screenshot
                        request_manual_screenshot("keyboard")

            # Get input
            get_joystick_input()

            # Clear screen
            screen.fill([0, 0, 0])

            # Get latest processed frame
            display_frame = None
            with data_lock:
                if current_processed_frame is not None:
                    display_frame = current_processed_frame.copy()
                    current_fps = fps
                    current_battery = battery_level
                    current_humans = humans_count
                    current_screenshots = screenshot_count
            
            
            if display_frame is not None:
                # Add info overlays
                cv2.putText(display_frame, f"Battery: {current_battery}%", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if current_humans > 0:
                    cv2.putText(display_frame, f"Humans: {current_humans}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.putText(display_frame, f"Screenshots: {current_screenshots}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                cv2.putText(display_frame, "4-THREAD MODE", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # Show countdown if active
                if countdown_active:
                    elapsed = time.time() - countdown_start_time
                    remaining = max(0, countdown_duration - elapsed)
                    cv2.putText(display_frame, f"Screenshot in: {remaining:.1f}s", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Convert frame for pygame display
                frame_rgb = np.rot90(display_frame)
                frame_rgb = np.flipud(frame_rgb)

                frame_surface = pygame.surfarray.make_surface(frame_rgb)
                screen.blit(frame_surface, (0, 0))

            pygame.display.update()
            time.sleep(1 / FPS)

    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    except Exception as e:
        print(f"Main loop error: {e}")
    finally:
        # Cleanup
        stop_threads()
        if tello:
            try:
                tello.streamoff()
                tello.end()
            except:
                pass
        if pose:
            pose.close()
        if hands:
            hands.close()
        pygame.quit()
        print(f"Done! Total screenshots taken: {screenshot_count}")


def main():
    """Main function"""
    print("4-Thread Tello Drone Control System")
    print("Architecture:")
    print("- Thread 1: Video stream, human detection, and screenshot processing")
    print("- Thread 2: Drone control commands and battery monitoring")
    print("- Thread 3: Red color detection")
    print("- Thread 4: Autonomous behavior")
    print("Features:")
    print("- Smart auto screenshot with 3-second countdown")
    print("- Manual screenshot with 'P' key or joystick buttons")
    print("- Human detection with pose and hand tracking")
    print("- Screenshots saved in 'screenshots' folder")
    print("Controls:")
    print("- Keyboard: Arrow keys=move, W/S=up/down, A/D=rotate, T=takeoff, L=land, P=screenshot")
    print("- Joystick: Move drone, A=takeoff, B=land, X/Y=screenshot")
    print("- ESC or close window to quit")
    
    try:
        # Initialize everything
        initialize_pygame()
        create_screenshot_directory()
        
        if not initialize_tello():
            print("Failed to initialize Tello. Exiting...")
            return
            
        if not initialize_ai_models():
            print("Failed to initialize AI models. Exiting...")
            return
        
        # Start video stream
        tello.streamoff()
        time.sleep(0.5)  # Brief pause
        tello.streamon()
        
        # Start worker threads
        start_threads()
        
        # Wait a moment for threads to initialize
        time.sleep(2)
        
        # Run main loop
        main_loop()
        
    except KeyboardInterrupt:
        print("Keyboard interrupt detected")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print("Shutting down...")


if __name__ == '__main__':
    main()