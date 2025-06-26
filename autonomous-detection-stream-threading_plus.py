import pygame
import numpy as np
import cv2
from djitellopy import Tello
import threading
import queue
import time

# Initialize pygame
pygame.init()
pygame.joystick.init()

# Set window size
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

# Create pygame window
screen = pygame.display.set_mode((WINDOW_WIDTH * 3, WINDOW_HEIGHT))
pygame.display.set_caption('Tello Red Detection - Original | Mask | Result')

# Initialize Tello
print("Connecting to Tello...")
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamon()
print("Video stream started...")

# Font for text
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)

# Global variables
latest_frame = None
movement_queue = queue.Queue()
command_queue = queue.Queue()
detection_results = queue.Queue()
set_autonomous_behavior = False

# Joystick variables
joystick = None
send_rc_control = False
left_right_velocity = 0
for_back_velocity = 0
up_down_velocity = 0
yaw_velocity = 0
last_joystick_screenshot_button_state = False

# Detection results storage
current_detection = {
    'red_detected': False,
    'mask': None,
    'result': None,
    'roi_mask': None,
    'full_roi_mask': None,
    'pixel_count': 0
}

# Thread-safe lock for shared variables
frame_lock = threading.Lock()
detection_lock = threading.Lock()

# ROI parameters (Region of Interest)
ROI_X = WINDOW_WIDTH // 4
ROI_Y = 5
ROI_WIDTH = WINDOW_WIDTH // 2
ROI_HEIGHT = WINDOW_HEIGHT // 3

def initialize_joystick():
    """Initialize joystick if available"""
    global joystick
    
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Joystick connected: {joystick.get_name()}")
        print(f"Number of axes: {joystick.get_numaxes()}")
        print(f"Number of buttons: {joystick.get_numbuttons()}")
        return True
    else:
        print("No joystick found")
        return False

def get_joystick_input():
    """Get joystick input and update velocities"""
    global left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity
    global send_rc_control, last_joystick_screenshot_button_state
    
    if not joystick:
        return

    speed = 50
    rotate = 80

    # Read joystick input
    axis_lr = joystick.get_axis(0)  # Left-right movement
    axis_fb = joystick.get_axis(1)  # Forward-backward movement
    axis_ud = joystick.get_axis(3)  # Up-down movement  
    axis_yv = joystick.get_axis(2)  # Yaw rotation

    # Apply deadzone to prevent drift
    deadzone = 0.1
    if abs(axis_lr) < deadzone:
        axis_lr = 0
    if abs(axis_fb) < deadzone:
        axis_fb = 0
    if abs(axis_ud) < deadzone:
        axis_ud = 0
    if abs(axis_yv) < deadzone:
        axis_yv = 0

    # Set velocities based on joystick input
    left_right_velocity = int(axis_lr * speed)
    for_back_velocity = int(-axis_fb * speed)
    up_down_velocity = int(-axis_ud * speed)
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
    
    # Additional buttons for drone control
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
    
    if joystick.get_button(6):  # Left trigger - enable autonomous behavior
        global set_autonomous_behavior
        set_autonomous_behavior = True
        print("🤖 Autonomous mode: ENABLED")
    
    if joystick.get_button(7):  # Right trigger - disable autonomous behavior
        set_autonomous_behavior = False
        print("🎮 Manual mode: ENABLED")

def request_manual_screenshot(source="manual"):
    """Request a screenshot"""
    try:
        with frame_lock:
            if latest_frame is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"tello_screenshot_{timestamp}_{source}.jpg"
                cv2.imwrite(filename, cv2.cvtColor(latest_frame, cv2.COLOR_RGB2BGR))
                print(f"📸 Screenshot saved: {filename}")
    except Exception as e:
        print(f"Screenshot error: {e}")

def detect_red_in_roi(img):
    """Detect red color specifically in ROI area"""
    if img is None:
        return False, None, None, 0
    
    bgr_img = img
    
    # Extract ROI
    roi = bgr_img[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH]
    
    # Detect red color in ROI
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 160, 130])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 130, 130])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
    mask_roi = cv2.bitwise_or(mask1, mask2)
    
    # Check if red color is detected
    red_detected = np.sum(mask_roi) > 1000
    pixel_count = np.sum(mask_roi > 0)
    
    # Create full-size mask for visualization
    full_mask = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=np.uint8)
    full_mask[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH] = mask_roi
    
    return red_detected, mask_roi, full_mask, pixel_count

def detect_red_color(img):
    """Detect red color using OpenCV for full image"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 180, 150])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 150])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    result = cv2.bitwise_and(img, img, mask=mask)
    return mask, result

def video_stream():
    """Thread for video streaming"""
    global latest_frame
    while True:
        try:
            frame_read = tello.get_frame_read()
            if frame_read.frame is not None:
                # Resize the frame and convert to RGB
                resized_frame = cv2.resize(frame_read.frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                
                # Thread-safe frame update
                with frame_lock:
                    latest_frame = rgb_frame
        except Exception as e:
            print(f"Video stream error: {e}")
            time.sleep(0.1)

def detection_thread():
    """Thread for red color detection"""
    global current_detection
    
    while True:
        try:
            # Get current frame safely
            with frame_lock:
                if latest_frame is not None:
                    frame_copy = latest_frame.copy()
                else:
                    frame_copy = None
            
            if frame_copy is not None:
                # Perform detection
                mask, result = detect_red_color(frame_copy)
                red_in_roi, roi_mask, full_roi_mask, pixel_count = detect_red_in_roi(frame_copy)
                
                # Update detection results safely
                with detection_lock:
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

def autonomous_behavior_thread():
    """Thread for autonomous behavior based on detection results"""
    while True:
        try:
            if set_autonomous_behavior:
                # Get current detection results safely
                with detection_lock:
                    red_detected = current_detection['red_detected']
                    pixel_count = current_detection['pixel_count']
                
                # Autonomous behavior logic
                if red_detected and pixel_count > 1000:
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

def rc_control_thread():
    """Thread for continuous RC control using joystick"""
    global send_rc_control
    
    while True:
        try:
            if send_rc_control and joystick:
                # Send RC control commands continuously
                tello.send_rc_control(left_right_velocity, for_back_velocity, 
                                    up_down_velocity, yaw_velocity)
                time.sleep(0.05)  # 20Hz update rate
            else:
                time.sleep(0.1)
        except Exception as e:
            print(f"RC control error: {e}")
            time.sleep(0.1)

def command_processor_thread():
    """Thread to process commands from queue"""
    while True:
        try:
            if not command_queue.empty():
                command = command_queue.get()
                
                if command == "takeoff":
                    print("🚁 Taking off...")
                    tello.takeoff()
                    send_rc_control = True
                elif command == "land":
                    print("🛬 Landing...")
                    send_rc_control = False
                    tello.land()
                elif command == "emergency":
                    print("🚨 Emergency stop!")
                    send_rc_control = False
                    tello.emergency()
                
                time.sleep(0.1)
            else:
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Command processor error: {e}")
            time.sleep(0.1)

def movement_control():
    """Thread for manual movement control (keyboard)"""
    while True:
        try:
            if not movement_queue.empty() and not set_autonomous_behavior and not send_rc_control:
                command = movement_queue.get()
                command_type, duration = command

                if command_type == "takeoff":
                    tello.takeoff()
                elif command_type == "land":
                    tello.land()
                elif command_type == "move_forward":
                    tello.move_forward(30)
                elif command_type == "move_back":
                    tello.move_back(30)
                elif command_type == "move_left":
                    tello.move_left(30)
                elif command_type == "move_right":
                    tello.move_right(30)
                elif command_type == "rotate_left":
                    tello.rotate_counter_clockwise(90)
                elif command_type == "rotate_right":
                    tello.rotate_clockwise(90)
                elif command_type == "move_up":
                    tello.move_up(30)
                elif command_type == "move_down":
                    tello.move_down(30)
                elif command_type == "emergency":
                    tello.emergency()

                time.sleep(duration)
            else:
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                
        except Exception as e:
            print(f"Movement control error: {e}")
            time.sleep(0.1)

def cv2_to_pygame(cv_img):
    """Convert OpenCV image to pygame surface"""
    if cv_img is None:
        return None
    
    if len(cv_img.shape) == 3:  # Color image
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    else:  # Grayscale image
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)

    rgb_img = np.transpose(rgb_img, (1, 0, 2))  # Transpose for pygame
    return pygame.surfarray.make_surface(rgb_img)

def draw_roi_rectangle(surface):
    """Draw ROI rectangle on pygame surface"""
    roi_color = (0, 255, 0)  # Green color for ROI
    roi_rect = pygame.Rect(ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT)
    pygame.draw.rect(surface, roi_color, roi_rect, 3)  # 3 pixel thick border

# Initialize joystick
joystick_available = initialize_joystick()

# Start all threads
print("Starting threads...")

# Video stream thread
thread_video = threading.Thread(target=video_stream, name="VideoStream")
thread_video.daemon = True
thread_video.start()

# Detection thread
thread_detection = threading.Thread(target=detection_thread, name="Detection")
thread_detection.daemon = True
thread_detection.start()

# Autonomous behavior thread
thread_autonomous = threading.Thread(target=autonomous_behavior_thread, name="Autonomous")
thread_autonomous.daemon = True
thread_autonomous.start()

# Movement control thread
thread_movement = threading.Thread(target=movement_control, name="Movement")
thread_movement.daemon = True
thread_movement.start()

# RC control thread for joystick
thread_rc = threading.Thread(target=rc_control_thread, name="RCControl")
thread_rc.daemon = True
thread_rc.start()

# Command processor thread
thread_commands = threading.Thread(target=command_processor_thread, name="Commands")
thread_commands.daemon = True
thread_commands.start()

print("All threads started successfully!")

# Main pygame loop
clock = pygame.time.Clock()
running = True

try:
    while running:
        # Process joystick input if available
        if joystick_available:
            pygame.event.pump()  # Update joystick state
            get_joystick_input()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_t:  # Takeoff
                    movement_queue.put(("takeoff", 5))
                elif event.key == pygame.K_l:  # Land
                    movement_queue.put(("land", 5))
                elif event.key == pygame.K_UP:  # Move forward
                    movement_queue.put(("move_forward", 1))
                elif event.key == pygame.K_DOWN:  # Move backward
                    movement_queue.put(("move_back", 1))
                elif event.key == pygame.K_LEFT:  # Move left
                    movement_queue.put(("move_left", 1))
                elif event.key == pygame.K_RIGHT:  # Move right
                    movement_queue.put(("move_right", 1))
                elif event.key == pygame.K_a:  # Rotate counter-clockwise
                    movement_queue.put(("rotate_left", 1))
                elif event.key == pygame.K_d:  # Rotate clockwise
                    movement_queue.put(("rotate_right", 1))
                elif event.key == pygame.K_SPACE:  # Enable autonomous behavior
                    set_autonomous_behavior = True
                    print("🤖 Autonomous mode: ENABLED")
                elif event.key == pygame.K_p:  # Disable autonomous behavior
                    set_autonomous_behavior = False
                    print("🎮 Manual mode: ENABLED")
                elif event.key == pygame.K_w:
                    movement_queue.put(('move_up', 1))
                elif event.key == pygame.K_s:
                    movement_queue.put(('move_down', 1))
                elif event.key == pygame.K_e:  # Emergency stop
                    movement_queue.put(("emergency", 0))
                elif event.key == pygame.K_c:  # Screenshot
                    request_manual_screenshot("keyboard")

        # Get current frame and detection results safely
        with frame_lock:
            current_frame = latest_frame.copy() if latest_frame is not None else None
        
        with detection_lock:
            detection_data = current_detection.copy()

        if current_frame is not None:
            # Convert images to pygame surface
            original_surface = cv2_to_pygame(current_frame)
            mask_surface = cv2_to_pygame(detection_data['mask'])
            result_surface = cv2_to_pygame(detection_data['result'])

            # Clear screen
            screen.fill((0, 0, 0))

            # Display the images
            if original_surface:
                screen.blit(original_surface, (0, 0))
            if mask_surface:
                screen.blit(mask_surface, (WINDOW_WIDTH, 0))
            if result_surface:
                screen.blit(result_surface, (WINDOW_WIDTH * 2, 0))

            # Draw ROI rectangle on original image
            draw_roi_rectangle(screen)

            # Add text labels
            original_text = font.render('Original + ROI', True, (255, 255, 255))
            mask_text = font.render('Red Mask', True, (255, 255, 255))
            result_text = font.render('Result', True, (255, 255, 255))
            screen.blit(original_text, (10, 10))
            screen.blit(mask_text, (WINDOW_WIDTH + 10, 10))
            screen.blit(result_text, (WINDOW_WIDTH * 2 + 10, 10))

            # Display battery status
            try:
                battery = tello.get_battery()
                battery_color = (0, 255, 0) if battery > 30 else (255, 255, 0) if battery > 15 else (255, 0, 0)
                battery_text = font.render(f'Battery: {battery}%', True, battery_color)
                screen.blit(battery_text, (10, 50))
            except:
                pass

            # Display control mode status
            if send_rc_control:
                mode_text = "JOYSTICK RC"
                mode_color = (0, 255, 255)
            elif set_autonomous_behavior:
                mode_text = "AUTO"
                mode_color = (0, 255, 0)
            else:
                mode_text = "MANUAL"
                mode_color = (255, 255, 0)
            
            mode_surface = font.render(f'Mode: {mode_text}', True, mode_color)
            screen.blit(mode_surface, (10, 80))

            # Display joystick status
            if joystick_available:
                joystick_status = f'Joystick: {joystick.get_name()[:20]}'
                joystick_surface = small_font.render(joystick_status, True, (0, 255, 255))
                screen.blit(joystick_surface, (10, 110))
                
                # Display joystick velocities when RC control is active
                if send_rc_control:
                    velocity_text = f'LR:{left_right_velocity} FB:{for_back_velocity} UD:{up_down_velocity} YAW:{yaw_velocity}'
                    velocity_surface = small_font.render(velocity_text, True, (255, 255, 255))
                    screen.blit(velocity_surface, (10, 130))

            # Display ROI detection status
            roi_status = "RED DETECTED" if detection_data['red_detected'] else "NO RED"
            roi_color = (255, 0, 0) if detection_data['red_detected'] else (128, 128, 128)
            roi_surface = font.render(f'ROI: {roi_status}', True, roi_color)
            screen.blit(roi_surface, (10, 150))

            # Display pixel count of red detection in ROI
            pixel_text = small_font.render(f'Red Pixels in ROI: {detection_data["pixel_count"]}', True, (255, 255, 255))
            screen.blit(pixel_text, (10, 170))

            # Control instructions
            instructions = [
                'Keyboard Controls:',
                'Q - Quit | T - Takeoff | L - Land',
                'Arrow Keys - Move | SPACE - Auto',
                'P - Manual | C - Screenshot',
                '',
                'Joystick Controls:',
                'A - Takeoff | B - Land',
                'LB - Enable RC | RB - Disable RC',
                'X/Y - Screenshot'
            ]
            
            y_offset = WINDOW_HEIGHT - 220
            for i, instruction in enumerate(instructions):
                if instruction == '':
                    continue
                color = (255, 255, 0) if 'Controls:' in instruction else (255, 255, 255)
                instruction_text = small_font.render(instruction, True, color)
                screen.blit(instruction_text, (10, y_offset + i * 20))

            pygame.display.flip()

        clock.tick(60)  # 60 FPS for smooth UI

except KeyboardInterrupt:
    print("Program stopped by user")
finally:
    # Cleanup
    print("Cleaning up resources...")
    running = False
    send_rc_control = False
    try:
        tello.streamoff()
        tello.end()
    except:
        pass
    
    if joystick:
        joystick.quit()
    
    pygame.quit()
    print("Program finished")