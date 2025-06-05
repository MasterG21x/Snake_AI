import pygame
from enum import Enum, auto

class Direction(Enum):
    """Enum representing the four possible snake directions."""
    RIGHT = auto()
    DOWN  = auto()
    LEFT  = auto()
    UP    = auto()

CLOCK_WISE = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

class Snake:
    """Main snake class"""

    def __init__(self, game):
        """
        Initializes the snake object with its head, segments, and direction.
        """
        self.screen = game.screen
        self.settings = game.settings

        self.snake_size = 20  
        self.alive = True

        start_x = self.settings.screen_width // 2
        start_y = self.settings.screen_height // 2

        self.segments = pygame.sprite.Group()
        self.head = SnakeSegment(game, start_x, start_y)
        self.segments.add(self.head)

        self.direction = Direction.RIGHT
        self.dx, self.dy = 1, 0
        self.position_history: list[tuple[int, int]] = []

        self.score = 0

    def move_once(self):
        """
        Moves the snake one step in the current direction.
        Updates position of the head and each body segment.
        Checks for collisions with walls.
        """
        self.dx, self.dy = {
            Direction.RIGHT: (1, 0),
            Direction.LEFT: (-1, 0),
            Direction.UP: (0, -1),
            Direction.DOWN: (0, 1),
        }[self.direction]

        x = self.head.rect.x + self.dx * self.snake_size
        y = self.head.rect.y + self.dy * self.snake_size

        self.head.rect.x, self.head.rect.y = x, y

        self.position_history.insert(0, (x, y))
        self.position_history = self.position_history[: len(self.segments)]

        for seg, pos in zip(self.segments.sprites()[1:], self.position_history[1:]):
            seg.rect.x, seg.rect.y = pos

        if (
            x < 0
            or x >= self.settings.screen_width
            or y < 0
            or y >= self.settings.screen_height
        ):
            self.alive = False

    def update_screen(self):
        """Drawing entire snake"""
        self.segments.draw(self.screen)

    def grow(self, game):
        """Snake segments adding while grow"""
        last = self.segments.sprites()[-1]
        x = last.rect.x - self.dx * self.snake_size
        y = last.rect.y - self.dy * self.snake_size
        self.segments.add(SnakeSegment(game, x, y))
            
class SnakeSegment(pygame.sprite.Sprite):
    """Snake body segments class"""
    def __init__(self, game, x, y):
        """Initialize single segment"""
        super().__init__()
        self.image = pygame.Surface((20, 20)) 
        self.image.fill((55, 160, 40))         
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
    