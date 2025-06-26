from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time
import sys
import mediapipe as mp
from ultralytics import YOLO
import os
from datetime import datetime
import threading
import queue
from collections import deque

# Global variables
S = 60  # Speed of the drone
FPS = 120  # Frames per second of the pygame window display

# Thread control
running = True
threads = []

# Thread-safe queues and locks
frame_queue = queue.Queue(maxsize=5)
processed_frame_queue = queue.Queue(maxsize=5)
command_queue = queue.Queue()
screenshot_queue = queue.Queue()

# Shared data with locks
data_lock = threading.Lock()
current_frame = None
current_processed_frame = None
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
    pygame.display.set_caption("Tello video stream (Threaded)")
    screen = pygame.display.set_mode([960, 720])
    
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
    global tello
    
    tello = Tello()
    tello.connect()
    tello.set_speed(speed)
    print(f"Battery: {tello.get_battery()}%")


def initialize_ai_models():
    """Initialize AI models for detection"""
    global yolo_model, mp_drawing, mp_drawing_styles, mp_pose, mp_hands, pose, hands
    
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


def create_screenshot_directory():
    """Create screenshots directory if it doesn't exist"""
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)

        print(f"Created directory: {screenshot_dir}")

def video_capture_thread():
    """Thread for capturing video frames from drone"""
    global running, current_frame
    
    print("Video capture thread started")
    frame_read = tello.get_frame_read()
    
    while running:
        try:
            if frame_read.stopped:
                break
                
            frame = frame_read.frame
            if frame is not None:
                # Resize frame once here
                frame = cv2.resize(frame, (960, 720))
                
                # Add to queue (non-blocking)
                if not frame_queue.full():
                    frame_queue.put(frame)
                
                # Update shared current frame
                with data_lock:
                    current_frame = frame.copy()
                    
            time.sleep(1/60)  # Limit to 60 FPS capture
            
        except Exception as e:
            print(f"Video capture error: {e}")
            break
    
    print("Video capture thread ended")


def process_human_detection(frame):
    """Process human detection and return processed frame with detection info"""
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


def handle_auto_screenshot(output_frame, human_detected_now, humans_count_now):
    """Handle auto screenshot countdown logic"""
    global countdown_active, countdown_start_time, last_human_detected
    
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
                screenshot_queue.put((output_frame.copy(), humans_count_now, "auto"))
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


def human_detection_thread():
    """Thread for processing human detection"""
    global running, current_processed_frame, human_detected, humans_count, fps, frame_times
    
    print("Human detection thread started")
    
    while running:
        try:
            # Get frame from queue
            frame = frame_queue.get(timeout=1)
            
            # Process human detection
            output_frame, detected, count = process_human_detection(frame)
            
            # Handle auto screenshot logic
            handle_auto_screenshot(output_frame, detected, count)
            
            # Update shared data
            with data_lock:
                current_processed_frame = output_frame.copy()
                human_detected = detected
                humans_count = count
                
                # Calculate FPS
                current_time = time.time()
                frame_times.append(current_time)
                if len(frame_times) > 1:
                    time_diff = frame_times[-1] - frame_times[0]
                    fps = len(frame_times) / time_diff if time_diff > 0 else 0
            
            # Add to processed frame queue
            if not processed_frame_queue.full():
                processed_frame_queue.put(output_frame)
            
            frame_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Human detection error: {e}")
            continue
    
    print("Human detection thread ended")


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


def drone_control_thread():
    """Thread for sending drone control commands"""
    global running, battery_level
    
    print("Drone control thread started")
    
    while running:
        try:
            # Process any queued commands
            while not command_queue.empty():
                command = command_queue.get_nowait()
                execute_drone_command(command)
            
            # Send RC control if active
            if send_rc_control:
                tello.send_rc_control(
                    left_right_velocity, 
                    for_back_velocity,
                    up_down_velocity, 
                    yaw_velocity
                )
            
            # Update battery level periodically
            if time.time() % 10 < 0.1:  # Every ~10 seconds
                try:
                    with data_lock:
                        battery_level = tello.get_battery()
                except:
                    pass
            
            time.sleep(1/30)  # 30 Hz control rate
            
        except Exception as e:
            print(f"Drone control error: {e}")
            time.sleep(0.1)
    
    print("Drone control thread ended")


def save_screenshot(frame, humans_count_param, source="auto"):
    """Save screenshot with timestamp and human count"""
    global screenshot_count
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_prefix = "manual" if source in ["joystick", "keyboard"] else "auto"
    filename = f"{source_prefix}_human_detected_{timestamp}_{humans_count_param}persons_{screenshot_count:04d}.jpg"
    filepath = os.path.join(screenshot_dir, filename)
    
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for saving

    # Save the frame
    success = cv2.imwrite(filepath, frame_bgr)
    
    if success:
        with data_lock:
            screenshot_count += 1
        print(f"Screenshot saved ({source}): {filename}")
        return True
    else:
        print(f"Failed to save screenshot: {filename}")
        return False


def screenshot_thread():
    """Thread for handling screenshot operations"""
    global running
    
    print("Screenshot thread started")
    
    while running:
        try:
            # Check for screenshot requests
            screenshot_data = screenshot_queue.get(timeout=1)
            frame, humans_count_param, source = screenshot_data
            
            save_screenshot(frame, humans_count_param, source)
            
            screenshot_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Screenshot error: {e}")
            continue
    
    print("Screenshot thread ended")


def get_joystick_input():
    """Get joystick input"""
    global left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity
    global last_joystick_screenshot_button_state
    
    if not joystick:
        return

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


def request_manual_screenshot(source):
    """Request a manual screenshot"""
    with data_lock:
        if current_processed_frame is not None:
            screenshot_queue.put((
                current_processed_frame.copy(), 
                humans_count, 
                source
            ))
            print(f"Manual screenshot requested ({source})")


def start_threads():
    """Start all worker threads"""
    global threads
    
    # Start video capture thread
    video_thread = threading.Thread(target=video_capture_thread, daemon=True)
    video_thread.start()
    threads.append(video_thread)
    
    # Start human detection thread
    detection_thread = threading.Thread(target=human_detection_thread, daemon=True)
    detection_thread.start()
    threads.append(detection_thread)
    
    # Start drone control thread
    control_thread = threading.Thread(target=drone_control_thread, daemon=True)
    control_thread.start()
    threads.append(control_thread)
    
    # Start screenshot thread
    screenshot_thread_obj = threading.Thread(target=screenshot_thread, daemon=True)
    screenshot_thread_obj.start()
    threads.append(screenshot_thread_obj)
    
    print(f"Started {len(threads)} worker threads")


def stop_threads():
    """Stop all worker threads"""
    global running
    
    print("Stopping threads...")
    running = False
    
    # Wait for threads to finish
    for thread in threads:
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
                    pass  # Update timer - not needed in threaded version
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True

            # Get joystick input
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
                
                cv2.putText(display_frame, "FUNCTION-BASED MODE", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # Convert frame for pygame display
                frame_rgb = np.rot90(display_frame)
                frame_rgb = np.flipud(frame_rgb)

                frame_surface = pygame.surfarray.make_surface(frame_rgb)
                screen.blit(frame_surface, (0, 0))

            pygame.display.update()
            time.sleep(1 / FPS)

    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    finally:
        # Cleanup
        stop_threads()
        tello.streamoff()
        tello.end()
        pose.close()
        hands.close()
        pygame.quit()
        print(f"Done! Total screenshots taken: {screenshot_count}")


def main():
    """Main function"""
    print("Function-based Tello Drone Control System")
    print("Features:")
    print("- Multi-threaded architecture for better performance")
    print("- Separate threads for video, detection, control, and screenshots")
    print("- Thread-safe communication between components")
    print("- Smart auto screenshot with 3-second countdown")
    print("- Manual screenshot with 'P' key or joystick buttons")
    print("- Screenshots saved in 'screenshots' folder")
    print("Controls:")
    print("- Keyboard: Arrow keys=move, W/S=up/down, A/D=rotate, T=takeoff, L=land, P=screenshot")
    print("- Joystick: Move drone, A=takeoff, B=land, X/Y=screenshot")
    print("- ESC or close window to quit")
    
    try:
        # Initialize everything
        initialize_pygame()
        create_screenshot_directory()
        initialize_tello()
        initialize_ai_models()
        
        # Start video stream
        tello.streamoff()
        tello.streamon()
        
        # Start worker threads
        start_threads()
        
        # Wait a moment for threads to initialize
        time.sleep(1)
        
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