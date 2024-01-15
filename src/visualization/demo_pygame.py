import pygame
import sys
import math

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
BIKE_SPEED = 5

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Set up the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bike Travel")

# Load bike image
bike_image = pygame.image.load("bike.jpeg")  # Replace 'bike.png' with your image file
bike_image = pygame.transform.scale(bike_image, (30, 30))  # Scale image if needed


# Bike position
bike_pos = [0, 0]


# Function to move the bike
def move_bike(destination):
    global bike_pos
    while bike_pos != destination:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Calculate movement direction
        direction = math.atan2(
            destination[1] - bike_pos[1], destination[0] - bike_pos[0]
        )
        bike_pos[0] += BIKE_SPEED * math.cos(direction)
        bike_pos[1] += BIKE_SPEED * math.sin(direction)

        # Round position to avoid overshooting
        bike_pos = [round(bike_pos[0]), round(bike_pos[1])]

        # Redraw
        screen.fill(WHITE)
        for i in range(0, WIDTH, 100):
            for j in range(0, HEIGHT, 100):
                pygame.draw.rect(screen, RED, (i, j, 20, 20))
        screen.blit(bike_image, bike_pos)
        pygame.display.flip()


# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            destination = [mouse_x // 100 * 100, mouse_y // 100 * 100]
            move_bike(destination)

    screen.fill(WHITE)
    for i in range(0, WIDTH, 100):
        for j in range(0, HEIGHT, 100):
            pygame.draw.rect(screen, RED, (i, j, 20, 20))
    screen.blit(bike_image, bike_pos)
    pygame.display.flip()

pygame.quit()
