import pygame
import numpy as np
import cv2
from djitellopy import Tello
import threading
import queue
import time

# Initialize pygame
pygame.init()

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

# Global variable for the latest frame
latest_frame = None
movement_queue = queue.Queue()

set_autonomous_behavior = False  # Flag for autonomous behavior

# ROI parameters (Region of Interest)
ROI_X = WINDOW_WIDTH // 4        # ROI start X (1/4 dari lebar)
ROI_Y = 5     # ROI start Y (1/4 dari tinggi)
ROI_WIDTH = WINDOW_WIDTH // 2    # ROI width (1/2 dari lebar)
ROI_HEIGHT = WINDOW_HEIGHT // 3  # ROI height (1/2 dari tinggi)

def detect_red_in_roi(img):
    """Detect red color specifically in ROI area"""
    if img is None:
        return False, None, None
    
    # Convert BGR to RGB for processing
    # bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bgr_img = img
    
    # Extract ROI
    roi = bgr_img[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH]
    
    # Detect red color in ROI
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 160, 130])   # Saturasi: 180→160 (-20), Brightness: 150→130 (-20)
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 130, 130]) # Saturasi: 150→130 (-20), Brightness: 150→130 (-20)
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
    mask_roi = cv2.bitwise_or(mask1, mask2)
    
    # Check if red color is detected (if there are white pixels in mask)
    red_detected = np.sum(mask_roi) > 25000  # Threshold untuk deteksi
    pixel = np.sum(mask_roi > 0)  # Hitung jumlah pixel yang terdeteksi merah
    
    # Create full-size mask for visualization
    full_mask = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=np.uint8)
    full_mask[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH] = mask_roi
    
    return red_detected, mask_roi, full_mask, pixel

def video_stream():
    global latest_frame
    while True:
        frame_read = tello.get_frame_read()
        if frame_read.frame is not None:
            # Resize the frame and convert to RGB
            resized_frame = cv2.resize(frame_read.frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            latest_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

def movement_control():
    global latest_frame
    while True:
        if not movement_queue.empty() and set_autonomous_behavior is False:
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

            time.sleep(duration)  # Sleep for the duration specifiedroi_mask

        if set_autonomous_behavior and latest_frame is not None:
            # Detect red color in ROI
            red_detected, roi_mask, full_mask, pixels = detect_red_in_roi(latest_frame)
            
            # IF-ELSE logic berdasarkan deteksi warna merah di ROI
            if red_detected:
                # JIKA terdeteksi warna merah di ROI
                print("🔴 Red detected in ROI! Moving towards target...")
                tello.move_back(50)  # Maju menuju target merah
                time.sleep(1)
                tello.rotate_clockwise(90)
                time.sleep(1)
            else:
                # JIKA TIDAK terdeteksi warna merah di ROI
                print("⚪ No red in ROI. Searching...")
                tello.move_forward(20)  # Putar untuk mencari target
                time.sleep(0.7)

# Start the video stream in a separate thread
thread_video = threading.Thread(target=video_stream)
thread_video.daemon = True
thread_video.start()

# Start the movement control in a separate thread
thread_movement = threading.Thread(target=movement_control)
thread_movement.daemon = True
thread_movement.start()

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

def cv2_to_pygame(cv_img):
    """Convert OpenCV image to pygame surface"""
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

# Main loop
clock = pygame.time.Clock()
running = True

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_t:  # Takeoff
                    movement_queue.put(("takeoff", 5))  # 5 seconds
                elif event.key == pygame.K_l:  # Land
                    movement_queue.put(("land", 5))  # 5 seconds
                elif event.key == pygame.K_UP:  # Move forward
                    movement_queue.put(("move_forward", 1))  # 1 second
                elif event.key == pygame.K_DOWN:  # Move backward
                    movement_queue.put(("move_back", 1))  # 1 second
                elif event.key == pygame.K_LEFT:  # Move left
                    movement_queue.put(("move_left", 1))  # 1 second
                elif event.key == pygame.K_RIGHT:  # Move right
                    movement_queue.put(("move_right", 1))  # 1 second
                elif event.key == pygame.K_a:  # Rotate counter-clockwise
                    movement_queue.put(("rotate_left", 1))  # 1 second
                elif event.key == pygame.K_d:  # Rotate clockwise
                    movement_queue.put(("rotate_right", 1))  # 1 second
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
                    movement_queue.put(("emergency", 0))  # Emergency stop

        if latest_frame is not None:  # Only process if the frame is available
            # Detect red color in full image
            mask, result = detect_red_color(latest_frame)
            
            # Detect red in ROI for autonomous behavior
            red_in_roi, roi_mask, full_roi_mask, pixel_red = detect_red_in_roi(latest_frame)

            # Convert images to pygame surface
            original_surface = cv2_to_pygame(latest_frame)
            mask_surface = cv2_to_pygame(mask)
            result_surface = cv2_to_pygame(result)

            # Clear screen
            screen.fill((0, 0, 0))

            # Display the images
            screen.blit(original_surface, (0, 0))  # Original
            screen.blit(mask_surface, (WINDOW_WIDTH, 0))  # Mask
            screen.blit(result_surface, (WINDOW_WIDTH * 2, 0))  # Result

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
                battery_text = font.render(f'Battery: {battery}%', True, (0, 255, 0))
                screen.blit(battery_text, (10, 50))
            except:
                pass

            # Display autonomous mode status
            mode_text = "AUTO" if set_autonomous_behavior else "MANUAL"
            mode_color = (0, 255, 0) if set_autonomous_behavior else (255, 255, 0)
            mode_surface = font.render(f'Mode: {mode_text}', True, mode_color)
            screen.blit(mode_surface, (10, 80))

            # Display ROI detection status
            roi_status = "RED DETECTED" if red_in_roi else "NO RED"
            roi_color = (255, 0, 0) if red_in_roi else (128, 128, 128)
            roi_surface = font.render(f'ROI: {roi_status}', True, roi_color)
            screen.blit(roi_surface, (10, 110))

            # Display pixel count of red detection in ROI
            pixel_text = font.render(f'Red Pixels in ROI: {pixel_red}', True, (255, 255, 255))
            screen.blit(pixel_text, (10, 140))

            # Control instructions
            instructions = [
                'Controls:',
                'Q - Quit',
                'T - Takeoff',
                'L - Land',
                'Arrow Keys - Move',
                'SPACE - Enable Auto',
                'P - Disable Auto'
            ]
            for i, instruction in enumerate(instructions):
                color = (255, 255, 0) if i == 0 else (255, 255, 255)
                instruction_text = font.render(instruction, True, color)
                screen.blit(instruction_text, (10, WINDOW_HEIGHT - 200 + i * 25))

            pygame.display.flip()

        clock.tick(30)  # 30 FPS

except KeyboardInterrupt:
    print("Program stopped by user")
finally:
    # Cleanup
    print("Cleaning up resources...")
    try:
        tello.streamoff()
        tello.end()
    except:
        pass
    pygame.quit()
    print("Program finished")