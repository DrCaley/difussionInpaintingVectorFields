import pygame
import numpy as np
import sys
import random
from scipy.fftpack import fft2, ifft2, fftfreq

# --- Divergence-Free Noise Generator ---
def generate_div_free_field(shape, scale=20.0):
    ny, nx = shape
    kx = fftfreq(nx) * nx / scale
    ky = fftfreq(ny) * ny / scale
    kx, ky = np.meshgrid(kx, ky)
    k_squared = kx ** 2 + ky ** 2
    k_squared[0, 0] = 1.0

    noise = np.random.randn(ny, nx) + 1j * np.random.randn(ny, nx)
    psi_hat = fft2(noise)
    u_hat = 1j * ky * psi_hat / k_squared
    v_hat = -1j * kx * psi_hat / k_squared
    u = np.real(ifft2(u_hat))
    v = np.real(ifft2(v_hat))
    return u, v

# --- Setup ---
pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Divergence-Free Platformer: Chaos v3")
clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 32)
big_font = pygame.font.SysFont(None, 48)

# Player
player = pygame.Rect(100, 100, 50, 50)
player_vel = [0.0, 0.0]
gravity = 0.5
on_ground = False
alive = True

# Platform
platform = pygame.Rect(0, HEIGHT - 50, WIDTH, 50)

# Goomba
goomba = pygame.Rect(WIDTH - 200, platform.top - 50, 50, 50)
goomba_alive = True
goomba_vel = -1.0

# Death messages
death_messages = [
    "You exploded randomly.",
    "Goomba wins this round.",
    "Noise giveth, noise taketh.",
    "The flow claimed you.",
    "Incompressible demise.",
    "You were swept away.",
    "You have been diverged.",
    "Pressure failure.",
    "Vector vortex of doom.",
    "Perished in the stream."
]
current_death_message = ""

# Divergence-free fields
noise_res = (64, 64)
u_field, v_field = generate_div_free_field(noise_res)
frame_count = 0
fps = 60
resize_timer = 0


def get_noise_value(x, y):
    xi = int(np.clip((x / WIDTH) * noise_res[1], 0, noise_res[1] - 1))
    yi = int(np.clip((y / HEIGHT) * noise_res[0], 0, noise_res[0] - 1))
    return u_field[yi, xi], v_field[yi, xi]


def reset():
    global player, player_vel, alive, on_ground, current_death_message, goomba_alive, goomba
    player.x, player.y = 100, 100
    player_vel = [0.0, 0.0]
    alive = True
    on_ground = False
    current_death_message = ""
    goomba_alive = True
    goomba.x = WIDTH - 200
    goomba.y = platform.top - 50


# Main loop
running = True
while running:
    frame_count += 1
    resize_timer += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Regenerate field every 90 frames
    if frame_count % 90 == 0:
        u_field, v_field = generate_div_free_field(noise_res)
        fps = int(np.clip(60 + np.mean(u_field) * 60, 15, 120))

    # Resize window randomly every 120 frames
    if resize_timer > 120:
        WIDTH = int(np.clip(800 + np.mean(v_field) * 600, 400, 1600))
        HEIGHT = int(np.clip(600 + np.mean(u_field) * 400, 300, 1000))
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        platform.width = WIDTH
        platform.y = HEIGHT - 50
        resize_timer = 0

    screen.fill((180, 200, 255))

    if alive:
        keys = pygame.key.get_pressed()
        move = 0
        if keys[pygame.K_LEFT]:
            move = -5
        elif keys[pygame.K_RIGHT]:
            move = 5

        u, v = get_noise_value(player.x, player.y)
        player_vel[0] = move + u * 5
        player_vel[1] += gravity + v * 2

        if keys[pygame.K_SPACE] and on_ground:
            player_vel[1] = -12 + v * 8
            on_ground = False

        if keys[pygame.K_d]:
            if np.abs(u + v) > 1.2:
                alive = False
                idx = int(np.clip(int(np.abs(u * 10)), 0, len(death_messages) - 1))
                current_death_message = death_messages[idx]

        # Apply movement
        player.x += int(player_vel[0])
        player.y += int(player_vel[1])

        if player.colliderect(platform):
            player.bottom = platform.top
            player_vel[1] = 0
            on_ground = True

        # Goomba behavior
        if goomba_alive:
            goomba.x += int(goomba_vel)
            if goomba.left <= 0 or goomba.right >= WIDTH:
                goomba_vel *= -1

            if player.colliderect(goomba):
                if player.bottom <= goomba.top + 10 and player_vel[1] > 0:
                    if np.abs(u + v) > 0.5:
                        goomba_alive = False
                        player_vel[1] = -8
                    else:
                        alive = False
                        idx = int(np.clip(int(np.abs(v * 10)), 0, len(death_messages) - 1))
                        current_death_message = death_messages[idx]
                else:
                    alive = False
                    idx = int(np.clip(int(np.abs(v * 10)), 0, len(death_messages) - 1))
                    current_death_message = death_messages[idx]

    else:
        death_text = big_font.render(current_death_message, True, (0, 0, 0))
        screen.blit(death_text, (WIDTH // 2 - death_text.get_width() // 2, HEIGHT // 2 - 30))

        restart_text = font.render("Press R to Restart", True, (0, 0, 0))
        screen.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2 + 20))

        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            reset()

    pygame.draw.rect(screen, (80, 50, 50), platform)
    if goomba_alive:
        pygame.draw.rect(screen, (180, 50, 50), goomba)
    if alive:
        pygame.draw.rect(screen, (50, 200, 50), player)

    # Overlay info
    vel_text = font.render(f"Field Vel: ({u:.2f}, {v:.2f})", True, (0, 0, 0))
    screen.blit(vel_text, (10, 10))

    fps_text = font.render(f"FPS: {fps}", True, (0, 0, 0))
    screen.blit(fps_text, (10, 40))

    screen_size_text = font.render(f"Size: {WIDTH}x{HEIGHT}", True, (0, 0, 0))
    screen.blit(screen_size_text, (10, 70))

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()
sys.exit()