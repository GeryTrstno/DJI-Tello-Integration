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

# Speed of the drone
S = 60
# Frames per second of the pygame window display
FPS = 120

class ThreadedFrontEnd(object):
    """ 
    Threaded Tello controller with separate threads for:
    - Video capture and processing
    - Human detection and screenshot logic
    - UI rendering and input handling
    - Drone control commands
    """

    def __init__(self):
        # Init pygame
        pygame.init()

        # Create pygame window
        pygame.display.set_caption("Tello video stream (Threaded)")
        self.screen = pygame.display.set_mode([960, 720])

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

        # Thread control
        self.running = True
        self.threads = []
        
        # Thread-safe queues and locks
        self.frame_queue = queue.Queue(maxsize=5)
        self.processed_frame_queue = queue.Queue(maxsize=5)
        self.command_queue = queue.Queue()
        self.screenshot_queue = queue.Queue()
        
        # Shared data with locks
        self.data_lock = threading.Lock()
        self.current_frame = None
        self.current_processed_frame = None
        self.battery_level = 0
        self.fps = 0
        self.humans_count = 0
        self.human_detected = False
        self.screenshot_count = 0
        
        # Performance monitoring
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()

        # Initialize joystick
        pygame.joystick.init()
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Joystick initialized: {self.joystick.get_name()}")
        else:
            print("No joystick detected - using keyboard only")

        # Create screenshots directory
        self.screenshot_dir = "screenshots"
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
            print(f"Created directory: {self.screenshot_dir}")

        # Load YOLOv8 model for human detection
        print("Loading YOLOv8 model...")
        self.yolo_model = YOLO('yolov8n.pt')

        # Mediapipe modules for body part detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands

        # Create MediaPipe models
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=1,  # Reduced for performance
            enable_segmentation=False,
            smooth_landmarks=True
        )

        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            max_num_hands=2
        )

        # Screenshot variables
        self.last_screenshot_time = 0
        self.screenshot_interval = 3

        # Auto screenshot countdown variables
        self.countdown_active = False
        self.countdown_start_time = 0
        self.countdown_duration = 3.0
        self.last_human_detected = False

        # Joystick screenshot variables
        self.last_joystick_screenshot_button_state = False
        self.joystick_screenshot_requested = False

        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

    def video_capture_thread(self):
        """Thread for capturing video frames from drone"""
        print("Video capture thread started")
        frame_read = self.tello.get_frame_read()
        
        while self.running:
            try:
                if frame_read.stopped:
                    break
                    
                frame = frame_read.frame
                if frame is not None:
                    # Resize frame once here
                    frame = cv2.resize(frame, (960, 720))
                    
                    # Add to queue (non-blocking)
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                    
                    # Update shared current frame
                    with self.data_lock:
                        self.current_frame = frame.copy()
                        
                time.sleep(1/60)  # Limit to 60 FPS capture
                
            except Exception as e:
                print(f"Video capture error: {e}")
                break
        
        print("Video capture thread ended")

    def human_detection_thread(self):
        """Thread for processing human detection"""
        print("Human detection thread started")
        
        while self.running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1)
                
                # Process human detection
                output_frame, human_detected, humans_count = self.process_human_detection(frame)
                
                # Handle auto screenshot logic
                self.handle_auto_screenshot(output_frame, human_detected, humans_count)
                
                # Update shared data
                with self.data_lock:
                    self.current_processed_frame = output_frame.copy()
                    self.human_detected = human_detected
                    self.humans_count = humans_count
                    
                    # Calculate FPS
                    current_time = time.time()
                    self.frame_times.append(current_time)
                    if len(self.frame_times) > 1:
                        time_diff = self.frame_times[-1] - self.frame_times[0]
                        self.fps = len(self.frame_times) / time_diff if time_diff > 0 else 0
                
                # Add to processed frame queue
                if not self.processed_frame_queue.full():
                    self.processed_frame_queue.put(output_frame)
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Human detection error: {e}")
                continue
        
        print("Human detection thread ended")

    def drone_control_thread(self):
        """Thread for sending drone control commands"""
        print("Drone control thread started")
        
        while self.running:
            try:
                # Process any queued commands
                while not self.command_queue.empty():
                    command = self.command_queue.get_nowait()
                    self.execute_drone_command(command)
                
                # Send RC control if active
                if self.send_rc_control:
                    self.tello.send_rc_control(
                        self.left_right_velocity, 
                        self.for_back_velocity,
                        self.up_down_velocity, 
                        self.yaw_velocity
                    )
                
                # Update battery level periodically
                if time.time() % 10 < 0.1:  # Every ~10 seconds
                    try:
                        with self.data_lock:
                            self.battery_level = self.tello.get_battery()
                    except:
                        pass
                
                time.sleep(1/30)  # 30 Hz control rate
                
            except Exception as e:
                print(f"Drone control error: {e}")
                time.sleep(0.1)
        
        print("Drone control thread ended")

    def screenshot_thread(self):
        """Thread for handling screenshot operations"""
        print("Screenshot thread started")
        
        while self.running:
            try:
                # Check for screenshot requests
                screenshot_data = self.screenshot_queue.get(timeout=1)
                frame, humans_count, source = screenshot_data
                
                self.save_screenshot(frame, humans_count, source)
                
                self.screenshot_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Screenshot error: {e}")
                continue
        
        print("Screenshot thread ended")

    def execute_drone_command(self, command):
        """Execute drone commands safely"""
        try:
            if command == "takeoff":
                self.tello.takeoff()
                self.send_rc_control = True
            elif command == "land":
                self.tello.land()
                self.send_rc_control = False
            elif command == "emergency":
                self.tello.emergency()
                self.send_rc_control = False
        except Exception as e:
            print(f"Command execution error: {e}")

    def save_screenshot(self, frame, humans_count, source="auto"):
        """Save screenshot with timestamp and human count"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_prefix = "manual" if source in ["joystick", "keyboard"] else "auto"
        filename = f"{source_prefix}_human_detected_{timestamp}_{humans_count}persons_{self.screenshot_count:04d}.jpg"
        filepath = os.path.join(self.screenshot_dir, filename)
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for saving

        # Save the frame
        success = cv2.imwrite(filepath, frame_bgr)
        
        if success:
            with self.data_lock:
                self.screenshot_count += 1
            print(f"Screenshot saved ({source}): {filename}")
            return True
        else:
            print(f"Failed to save screenshot: {filename}")
            return False

    def get_joystick_input(self):
        """Get joystick input (called from main thread)"""
        if not self.joystick:
            return

        speed = 50
        rotate = 80

        # Read joystick input
        axis_lr = self.joystick.get_axis(0)  # Left-right movement
        axis_fb = self.joystick.get_axis(1)  # Forward-backward movement
        axis_yv = self.joystick.get_axis(2)  # Up-down movement  
        axis_ud = self.joystick.get_axis(3)  # Yaw rotation

        # Set velocities based on joystick input
        self.left_right_velocity = int(axis_lr * speed)
        self.for_back_velocity = int(-axis_fb * speed)
        self.up_down_velocity = int(-axis_ud * speed)
        self.yaw_velocity = int(axis_yv * rotate)

        # Handle buttons
        if self.joystick.get_button(0):  # Button A - takeoff
            if not self.send_rc_control:
                self.command_queue.put("takeoff")
                time.sleep(0.5)

        if self.joystick.get_button(1):  # Button B - land
            if self.send_rc_control:
                self.command_queue.put("land")
                time.sleep(0.5)

        # Screenshot buttons
        current_screenshot_button_state = self.joystick.get_button(2)
        if current_screenshot_button_state and not self.last_joystick_screenshot_button_state:
            self.request_manual_screenshot("joystick")
        
        self.last_joystick_screenshot_button_state = current_screenshot_button_state

        if self.joystick.get_button(3):  # Alternative screenshot button
            self.request_manual_screenshot("joystick")
            time.sleep(0.2)

    def request_manual_screenshot(self, source):
        """Request a manual screenshot"""
        with self.data_lock:
            if self.current_processed_frame is not None:
                self.screenshot_queue.put((
                    self.current_processed_frame.copy(), 
                    self.humans_count, 
                    source
                ))
                print(f"Manual screenshot requested ({source})")

    def process_human_detection(self, frame):
        """Process human detection and return processed frame with detection info"""
        # Make a copy for processing
        output_frame = frame.copy()
        
        # YOLOv8 Human Detection
        results = self.yolo_model(frame, verbose=False)

        human_detected = False
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
                        human_detected = True

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
        if human_detected:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with pose detection
            pose_results = self.pose.process(rgb_frame)

            # Process with hand detection
            hands_results = self.hands.process(rgb_frame)

            # Draw hands if detected
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        output_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Draw pose landmarks if detected
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    output_frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )

        return output_frame, human_detected, len(human_boxes)

    def handle_auto_screenshot(self, output_frame, human_detected, humans_count):
        """Handle auto screenshot countdown logic"""
        current_time = time.time()
        
        if human_detected and humans_count >= 1:
            # Human detected
            if not self.last_human_detected and not self.countdown_active:
                # First time detecting human, start countdown
                self.countdown_active = True
                self.countdown_start_time = current_time
                print(f"Human detected! Starting 3-second countdown...")
            
            # If countdown is active
            if self.countdown_active:
                elapsed_time = current_time - self.countdown_start_time
                
                if elapsed_time >= self.countdown_duration:
                    # Countdown finished, take screenshot
                    self.screenshot_queue.put((output_frame.copy(), humans_count, "auto"))
                    self.last_screenshot_time = current_time
                    self.countdown_active = False
                    print("Countdown completed! Screenshot taken.")
        else:
            # No human detected
            if self.countdown_active:
                # Cancel countdown if human disappears
                self.countdown_active = False
                print("Human detection lost! Countdown cancelled.")
        
        # Update human detection status for next frame
        self.last_human_detected = human_detected

    def start_threads(self):
        """Start all worker threads"""
        # Start video capture thread
        video_thread = threading.Thread(target=self.video_capture_thread, daemon=True)
        video_thread.start()
        self.threads.append(video_thread)
        
        # Start human detection thread
        detection_thread = threading.Thread(target=self.human_detection_thread, daemon=True)
        detection_thread.start()
        self.threads.append(detection_thread)
        
        # Start drone control thread
        control_thread = threading.Thread(target=self.drone_control_thread, daemon=True)
        control_thread.start()
        self.threads.append(control_thread)
        
        # Start screenshot thread
        screenshot_thread = threading.Thread(target=self.screenshot_thread, daemon=True)
        screenshot_thread.start()
        self.threads.append(screenshot_thread)
        
        print(f"Started {len(self.threads)} worker threads")

    def stop_threads(self):
        """Stop all worker threads"""
        print("Stopping threads...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2)
        
        print("All threads stopped")

    def run(self):
        """Main run loop - handles UI and input"""
        # Connect to drone
        self.tello.connect()
        self.tello.set_speed(self.speed)
        print(f"Battery: {self.tello.get_battery()}%")

        # Start video stream
        self.tello.streamoff()
        self.tello.streamon()
        
        # Start worker threads
        self.start_threads()
        
        # Wait a moment for threads to initialize
        time.sleep(1)

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
                        else:
                            self.keydown(event.key)
                    elif event.type == pygame.KEYUP:
                        self.keyup(event.key)

                # Get joystick input
                self.get_joystick_input()

                # Clear screen
                self.screen.fill([0, 0, 0])

                # Get latest processed frame
                display_frame = None
                with self.data_lock:
                    if self.current_processed_frame is not None:
                        display_frame = self.current_processed_frame.copy()
                        current_fps = self.fps
                        current_battery = self.battery_level
                        current_humans = self.humans_count
                        current_screenshots = self.screenshot_count

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
                    
                    cv2.putText(display_frame, "THREADED MODE", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    # Convert frame for pygame display
                    frame_rgb = np.rot90(display_frame)
                    frame_rgb = np.flipud(frame_rgb)

                    frame_surface = pygame.surfarray.make_surface(frame_rgb)
                    self.screen.blit(frame_surface, (0, 0))

                pygame.display.update()
                time.sleep(1 / FPS)

        except KeyboardInterrupt:
            print("Keyboard interrupt received")
        finally:
            # Cleanup
            self.stop_threads()
            self.tello.streamoff()
            self.tello.end()
            self.pose.close()
            self.hands.close()
            pygame.quit()
            print(f"Done! Total screenshots taken: {self.screenshot_count}")

    def keydown(self, key):
        """Update velocities based on key pressed"""
        if key == pygame.K_w:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_s:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_a:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_d:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_UP:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_p:  # Manual screenshot
            self.request_manual_screenshot("keyboard")
        elif key == pygame.K_DOWN:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_LEFT:  # set yaw counter clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_RIGHT:  # set yaw clockwise velocity
            self.yaw_velocity = S
        elif key == pygame.K_q:  # Quit
            self.running = False
            pygame.quit()
            sys.exit()

    def keyup(self, key):
        """Update velocities based on key released"""
        if key == pygame.K_w or key == pygame.K_s:
            self.for_back_velocity = 0
        elif key == pygame.K_d or key == pygame.K_a:
            self.left_right_velocity = 0
        elif key == pygame.K_UP or key == pygame.K_DOWN:
            self.up_down_velocity = 0
        elif key == pygame.K_RIGHT or key == pygame.K_LEFT:
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.command_queue.put("takeoff")
        elif key == pygame.K_l:  # land
            self.command_queue.put("land")


def main():
    frontend = ThreadedFrontEnd()
    
    print("Threaded Tello Drone Control System")
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
        frontend.run()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print("Shutting down...")


if __name__ == '__main__':
    main()