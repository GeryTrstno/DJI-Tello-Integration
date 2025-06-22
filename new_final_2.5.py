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

# Speed of the drone
S = 60
# Frames per second of the pygame window display
FPS = 120

class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys and joystick.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations (yaw)
            - W and S: Up and down.
            - S: Manual screenshot
            - Joystick: A=takeoff, B=land, X/Y=screenshot
    """

    def __init__(self):
        # Init pygame
        pygame.init()

        # Create pygame window
        pygame.display.set_caption("Tello video stream")
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
            model_complexity=2,
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
        self.screenshot_count = 0

        # Auto screenshot countdown variables
        self.countdown_active = False
        self.countdown_start_time = 0
        self.countdown_duration = 3.0
        self.last_human_detected = False

        # Joystick screenshot variables
        self.last_joystick_screenshot_button_state = False
        self.joystick_screenshot_requested = False

        # FPS variables
        self.prev_time = time.time()
        self.fps = 0

        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

    def save_screenshot(self, frame, humans_count, source="auto"):
        """Save screenshot with timestamp and human count"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_prefix = "manual" if source in ["joystick", "keyboard"] else "auto"
        filename = f"{source_prefix}_human_detected_{timestamp}_{humans_count}persons_{self.screenshot_count:04d}.jpg"
        filepath = os.path.join(self.screenshot_dir, filename)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Save the frame
        success = cv2.imwrite(filepath, frame)
        
        if success:
            self.screenshot_count += 1
            print(f"Screenshot saved ({source}): {filename}")
            return True
        else:
            print(f"Failed to save screenshot: {filename}")
            return False

    def get_joystick_input(self):
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
                self.tello.takeoff()
                self.send_rc_control = True
                time.sleep(0.5)

        if self.joystick.get_button(1):  # Button B - land
            if self.send_rc_control:
                self.tello.land()
                self.send_rc_control = False
                time.sleep(0.5)

        # Screenshot buttons
        current_screenshot_button_state = self.joystick.get_button(2)
        if current_screenshot_button_state and not self.last_joystick_screenshot_button_state:
            self.joystick_screenshot_requested = True
            print("Joystick screenshot button pressed!")
        
        self.last_joystick_screenshot_button_state = current_screenshot_button_state

        if self.joystick.get_button(3):  # Alternative screenshot button
            self.joystick_screenshot_requested = True
            print("Alternative joystick screenshot button pressed!")
            time.sleep(0.2)

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
                        confidence_percentage = confidence * 100  # Convert to percentage
                        label = f"Human: {confidence_percentage:.0f}%"  # Format as an integer percentage

                        cv2.putText(output_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Process detailed body part detection if human detected
        if human_detected:

            # Process with pose detection
            pose_results = self.pose.process(frame)

            # Process with hand detection
            hands_results = self.hands.process(frame)

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
                remaining_time = self.countdown_duration - elapsed_time
                
                if elapsed_time >= self.countdown_duration:
                    # Countdown finished, take screenshot
                    self.save_screenshot(output_frame, humans_count, "auto")
                    self.last_screenshot_time = current_time
                    self.countdown_active = False
                    print("Countdown completed! Screenshot taken.")
                else:
                    # Show countdown with progress bar
                    progress = elapsed_time / self.countdown_duration
                    
                    # Main countdown text
                    # cv2.putText(output_frame, f"COUNTDOWN: {remaining_time:.1f}s", (10, 150),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    # cv2.putText(output_frame, "Keep human in view...", (10, 180),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Progress bar
                    bar_width = 200
                    bar_height = 10
                    bar_x = 10
                    bar_y = 200
                    
                    # Background bar
                    # cv2.rectangle(output_frame, (bar_x, bar_y), 
                    #               (bar_x + bar_width, bar_y + bar_height), (128, 128, 128), -1)
                    
                    # # Progress bar with color gradient
                    # filled_width = int(bar_width * progress)
                    # if progress < 0.5:
                    #     color = (0, 255, int(255 * progress * 2))
                    # else:
                    #     color = (0, int(255 * (2 - progress * 2)), 255)
                    
                    # cv2.rectangle(output_frame, (bar_x, bar_y), 
                    #               (bar_x + filled_width, bar_y + bar_height), color, -1)
                    
                    # # Progress bar border
                    # cv2.rectangle(output_frame, (bar_x, bar_y), 
                    #               (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        else:
            # No human detected
            if self.countdown_active:
                # Cancel countdown if human disappears
                self.countdown_active = False
                print("Human detection lost! Countdown cancelled.")
        
        # Update human detection status for next frame
        self.last_human_detected = human_detected

    def run(self):
        self.tello.connect()
        self.tello.set_speed(self.speed)
        print(f"Battery: {self.tello.get_battery()}%")

        # In case streaming is on
        self.tello.streamoff()
        self.tello.streamon()

        frame_read = self.tello.get_frame_read()

        should_stop = False
        while not should_stop:

            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)

            if frame_read.stopped:
                break

            # Get joystick input
            self.get_joystick_input()

            self.screen.fill([0, 0, 0])

            frame = frame_read.frame
            if frame is None:
                continue

            # Calculate FPS
            curr_time = time.time()
            self.fps = 1 / (curr_time - self.prev_time) if curr_time != self.prev_time else 0
            self.prev_time = curr_time

            # Resize frame for processing
            frame = cv2.resize(frame, (960, 720))

            # Process human detection
            output_frame, human_detected, humans_count = self.process_human_detection(frame)

            # Handle auto screenshot logic  
            self.handle_auto_screenshot(output_frame, human_detected, humans_count)

            # # Handle joystick screenshot request
            # if self.joystick_screenshot_requested:
            #     self.save_screenshot(output_frame, humans_count, "joystick")
            #     cv2.putText(output_frame, "JOYSTICK SCREENSHOT SAVED!", (10, 250),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            #     self.joystick_screenshot_requested = False

            # # Add info overlays
            # cv2.putText(output_frame, f"Battery: {self.tello.get_battery()}%", (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(output_frame, f"FPS: {self.fps:.1f}", (10, 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # if humans_count > 0:
            #     cv2.putText(output_frame, f"Humans Detected: {humans_count}", (10, 90),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # cv2.putText(output_frame, f"Screenshots: {self.screenshot_count}", (10, 120),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Convert frame for pygame display
            frame_rgb = np.rot90(output_frame)
            frame_rgb = np.flipud(frame_rgb)

            frame_surface = pygame.surfarray.make_surface(frame_rgb)
            self.screen.blit(frame_surface, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)

        # Cleanup
        self.tello.streamoff()
        self.tello.end()
        self.pose.close()
        self.hands.close()
        print(f"Done! Total screenshots taken: {self.screenshot_count}")

    def keydown(self, key):
        """ Update velocities based on key pressed """
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
        elif key == pygame.K_p:  # Manual screenshot or set down velocity
            # Check if 's' is pressed alone for screenshot
            keys = pygame.key.get_pressed()
            if not any([keys[pygame.K_LSHIFT], keys[pygame.K_RSHIFT], 
                       keys[pygame.K_LCTRL], keys[pygame.K_RCTRL]]):
                # Take screenshot
                frame = self.tello.get_frame_read().frame
                if frame is not None:
                    frame = cv2.resize(frame, (960, 720))
                    output_frame, _, humans_count = self.process_human_detection(frame)
                    self.save_screenshot(output_frame, humans_count, "keyboard")
                    print("Manual keyboard screenshot taken!")
        elif key == pygame.K_DOWN:  # set down velocity
                self.up_down_velocity = -S
        elif key == pygame.K_LEFT:  # set yaw counter clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_RIGHT:  # set yaw clockwise velocity
            self.yaw_velocity = S
        elif key == pygame.K_q:
            """ Quit the program """
            pygame.quit()
            sys.exit()
            self.tello.streamoff()
            self.tello.end()
            self.pose.close()
            self.hands.close()
            print(f"Done! Total screenshots taken: {self.screenshot_count}")
            sys.exit()

    def keyup(self, key):
        """ Update velocities based on key released """
        if key == pygame.K_w or key == pygame.K_s:
            self.for_back_velocity = 0
        elif key == pygame.K_d or key == pygame.K_a:
            self.left_right_velocity = 0
        elif key == pygame.K_UP or key == pygame.K_DOWN:
            self.up_down_velocity = 0
        elif key == pygame.K_RIGHT or key == pygame.K_LEFT:
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False

    def update(self):
        """ Update routine. Send velocities to Tello. """
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)


def main():
    frontend = FrontEnd()
    
    print("Tello Drone Control with Keyboard/Joystick, Human Detection and Smart Auto Screenshot")
    print("Features:")
    print("- Smart auto screenshot: 3-second countdown after human detection")
    print("- Countdown cancels if human detection is lost")
    print("- Manual screenshot with 's' key")
    print("- Manual screenshot with joystick buttons")
    print("- Screenshots saved in 'screenshots' folder")
    print("Controls:")
    print("- Keyboard: Arrow keys=move, W/S=up/down, A/D=rotate, T=takeoff, L=land, S=screenshot")
    print("- Joystick: Move drone, A=takeoff, B=land, X/Y=screenshot")
    print("- ESC or close window to quit")
    
    try:
        frontend.run()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected")
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == '__main__':
    main()