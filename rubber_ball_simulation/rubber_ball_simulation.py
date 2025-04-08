import pygame
import cv2
import numpy as np
import imageio

pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
R = 100  # Radius of the rolling object
M = 50000  # Mass of the wall
m = 1  # Mass of the rolling object
I = 2 / 3 * m * R ** 2  # Moment of inertia 

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
LIGHT_RED = (255, 100, 100)

# Setup display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rolling Object Collision Simulation")

# Rolling object properties
v_0 = 150  # Initial linear velocity (increased for better speed)
omega_0 = -v_0 / R  # Initial angular velocity
x = WIDTH - 100  # Initial position of the object, near the right side
y = HEIGHT - R  # Object on the ground
vx = -v_0  # Initial horizontal velocity (pointing to the left)
vy = 0  # Initial vertical velocity
omega = omega_0  # Initial angular velocity

# Rotation visualization
theta = 0.0  # Initial angle in radians
marker_length = R - 3  # Length of the marker on the circumference

# Wall properties (left side of the screen)
wall_x = 100  # Wall is at x = 100
wall_y = HEIGHT - R  # Same y position as the object
wall_vx = 0  # Wall is stationary
wall_vy = 0  # Wall has no vertical velocity

# Set up video writer (MP4 format)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
video_writer = cv2.VideoWriter('shell_animation.mp4', fourcc, FPS / 
                               1, (WIDTH, HEIGHT))
gif_writer = imageio.get_writer('rolling_object_animation.gif', duration=1/FPS)  # Match GIF speed to simulation speed


# Simulation loop
clock = pygame.time.Clock()
running = True
after_collision = False

while running:
    screen.fill(WHITE)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move the rolling object
    x += vx / FPS 
    y += -vy / FPS

    # Update angular velocity based on translational velocity (v_0 = R * omega)
    if after_collision == False:
        omega = vx / R 
    elif after_collision == True:
        omega = -vy / R  # Angular velocity for rolling after collision

    theta += omega / FPS  # Update the rotation angle (radians)

    # Check for collision with the wall
    if x - R < wall_x and after_collision == False:  # Detect collision with the left wall
        after_collision = True
        
        # Elastic collision in the x-direction (conserving linear momentum)
        v_prime_x = -(m - M) / (m + M) * v_0 - (2 * M) / (m + M) * wall_vx
        vx = v_prime_x  # Update the velocity of the rolling object
        # Update vertical velocity (vy) after collision using the formula
        v_prime_y = (I / (I + m * R ** 2)) * v_0
        vy = v_prime_y  # Update the vertical velocity

    # Stop the simulation if the ball leaves the screen (right side)
    if x + R > WIDTH:
        running = False

    # Ball center
    cx, cy = int(x), int(y)

    # Marker point on the circumference
    mx = cx - marker_length * np.cos(theta)
    my = cy - marker_length * np.sin(theta)  # subtract because y is upward

    # Draw the rolling object
    pygame.draw.circle(screen, BLUE, (cx, cy), R)  # Ball
    pygame.draw.circle(screen, LIGHT_RED, (int(mx), int(my)), 5)  # Rolling marker

    # Draw the wall (a simple line for simplicity)
    pygame.draw.rect(screen, BLACK, (wall_x - 10, 0, 10, HEIGHT))  # Wall on left side

    # Convert the screen to a frame and write it to the video
    frame = np.array(pygame.surfarray.pixels3d(screen))

    # Flip the frame for correct orientation
    frame = np.fliplr(frame)

    # Rotate the frame by 90 degrees counterclockwise
    frame = np.rot90(frame)


    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Write the frame to the video
    video_writer.write(frame_bgr)
    gif_writer.append_data(frame)

    pygame.display.flip()

    clock.tick(FPS)


# Release the video writer
video_writer.release()
gif_writer.close()

pygame.quit()
