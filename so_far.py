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

def video_stream():
    global latest_frame
    while True:
        frame_read = tello.get_frame_read()
        if frame_read.frame is not None:
            # Resize the frame and convert to RGB
            resized_frame = cv2.resize(frame_read.frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            latest_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

def movement_control():
    while True:
        if not movement_queue.empty():
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

            time.sleep(duration)  # Sleep for the duration specified

# Start the video stream in a separate thread
thread_video = threading.Thread(target=video_stream)
thread_video.daemon = True
thread_video.start()

# Start the movement control in a separate thread
thread_movement = threading.Thread(target=movement_control)
thread_movement.daemon = True
thread_movement.start()

def detect_red_color(img):
    """Detect red color using OpenCV"""
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

        if latest_frame is not None:  # Only process if the frame is available
            # Detect red color
            mask, result = detect_red_color(latest_frame)

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

            # Add text labels
            original_text = font.render('Original', True, (255, 255, 255))
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

            # Control instructions
            instructions = [
                'Controls:',
                'Q - Quit',
                'T - Takeoff',
                'L - Land',
                'Arrow Keys - Move'
            ]
            for i, instruction in enumerate(instructions):
                color = (255, 255, 0) if i == 0 else (255, 255, 255)
                instruction_text = font.render(instruction, True, color)
                screen.blit(instruction_text, (10, WINDOW_HEIGHT - 150 + i * 25))

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