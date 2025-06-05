import pygame 
from pygame.sprite import Sprite
import random

class Food(Sprite):
    """Food class"""
    def __init__(self, snake_game):
        """Initialize food object"""
        super().__init__() 
        self.screen = snake_game.screen
        self.settings = snake_game.settings

        self.size = 20
        self.color = (0, 0, 0)
        self.rect = pygame.Rect(0, 0, self.size, self.size)

        self.reposition() 

    def reposition(self):
        """Food positioning"""
        max_x = self.settings.screen_width - self.size
        max_y = self.settings.screen_height - self.size

        self.rect.x = random.randint(0, max_x // self.size) * self.size
        self.rect.y = random.randint(0, max_y // self.size) * self.size

    def draw_food(self):
        """Food drawing"""
        pygame.draw.rect(self.screen, self.color, self.rect)