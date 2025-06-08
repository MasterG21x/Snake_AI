import sys
import numpy as np
import pygame

from snake import Snake, Direction, CLOCK_WISE
from food import Food
from settings import Settings


class Board:
    """Main game class"""

    def __init__(self, headless: bool = False):
        """
        Initializes the game board, snake, and food.
        headless - disables graphical display (useful for AI training).
        """
        pygame.init()
        self.settings = Settings()
        self.headless = headless
        
        if not headless:
            self.screen = pygame.display.set_mode(
                (self.settings.screen_width, self.settings.screen_height), pygame.NOFRAME
            )
            pygame.display.set_caption("Snake")
        else:
            self.screen = pygame.Surface(
                (self.settings.screen_width, self.settings.screen_height)
            )

        self.font = pygame.font.SysFont(None, 30)
        self.clock = pygame.time.Clock()

        self.snake = Snake(self)
        self.foods = pygame.sprite.Group()
        self._draw_food()

    
    def run_game(self):
        "Manual game loop"
        while True:
            self._check_events()

            self.snake.move_once()
            self._post_move_updates()

            if not self.headless:
                self._update_screen()

            if not self.snake.alive:
                self.reset()

            self.clock.tick(10) 

    def _check_events(self):
        """Handles user input and system events (keyboard and window closing)."""
        if self.headless:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                self._keydown(event)

    def _keydown(self, event: pygame.event.Event):
        cur = self.snake.direction

        if event.key == pygame.K_LEFT and cur != Direction.RIGHT:
            self.snake.direction = Direction.LEFT
        elif event.key == pygame.K_RIGHT and cur != Direction.LEFT:
            self.snake.direction = Direction.RIGHT
        elif event.key == pygame.K_UP and cur != Direction.DOWN:
            self.snake.direction = Direction.UP
        elif event.key == pygame.K_DOWN and cur != Direction.UP:
            self.snake.direction = Direction.DOWN
        elif event.key == pygame.K_q:
            pygame.quit()
            sys.exit()

    def _update_screen(self):
        """Updates the entire game screen."""
        self.screen.fill(self.settings.bg_color)
        self._draw_score()
        self.snake.update_screen()
        for f in self.foods:
            f.draw_food()
        pygame.display.flip()

    def _draw_score(self):
        """Renders the current score on the screen."""
        txt = self.font.render(f"Score: {self.snake.score}", True, (0, 0, 0))
        self.screen.blit(txt, (self.settings.screen_width - 120, 10))

    def _post_move_updates(self):
        """Handles updates after the snake moves"""
        ate = self._check_food_collision()
        self._check_self_collision()
        if ate or len(self.foods) == 0:
            self._draw_food()

    def _draw_food(self):
        """Places new food on the board."""
        while len(self.foods) < self.settings.food_allowed:
            f = Food(self)
            if not pygame.sprite.spritecollideany(f, self.snake.segments):
                self.foods.add(f)
            else:
                f.kill()

    def _check_food_collision(self) -> bool:
        """Food collision check"""
        ate = False
        hits = pygame.sprite.groupcollide(self.foods, self.snake.segments, True, False)
        if hits:
            for _ in hits:
                ate = True
                self.snake.score += 1
                self.snake.grow(self)
        return ate

    def _check_self_collision(self):
        """Body collision check"""
        head, *body = self.snake.segments.sprites()
        for seg in body:
            if head.rect.colliderect(seg.rect):
                self.snake.alive = False
                break

    def reset(self):
        """Game reset"""
        self.snake = Snake(self)
        self.foods.empty()
        self._draw_food()
    
    def render(self) -> None:
        """Learning visualisation"""
        if self.headless:
            return
        self._update_screen()          
    
    def play_step(self, action: int):
        """Performs one step of the game using the specified action.
        Used by agent to control the snake. """
        
        idx = CLOCK_WISE.index(self.snake.direction)
        if action == 1:
            new_dir = CLOCK_WISE[(idx + 1) % 4]
        elif action == 2:
            new_dir = CLOCK_WISE[(idx - 1) % 4]
        else:
            new_dir = self.snake.direction

        if new_dir != {Direction.RIGHT: Direction.LEFT,
                       Direction.LEFT: Direction.RIGHT,
                       Direction.UP: Direction.DOWN,
                       Direction.DOWN: Direction.UP}[self.snake.direction]:
            self.snake.direction = new_dir

        score_before = self.snake.score
        self.snake.move_once()
        self._post_move_updates()

        reward = -0.1  
        done = not self.snake.alive

        if self.snake.score > score_before:
            reward = 10.0
        elif done:
            reward = -10.0
        if self.head_is_surrounded():
            reward -= 0.2
        return reward, done, self.snake.score

    
    def _point_collision(self, x: int, y: int) -> bool:
        """Helper for DQN, wall collision check"""
        if x < 0 or x >= self.settings.screen_width or y < 0 or y >= self.settings.screen_height:
            return True
        return any(seg.rect.x == x and seg.rect.y == y for seg in self.snake.segments)

    def get_state(self) -> np.ndarray:
        """
        Returns the current state vector for learning.
        Includes info about dangers, movement direction, and food location.
        """
        head = self.snake.head
        dir_x, dir_y = self.snake.dx, self.snake.dy
        size = self.snake.snake_size

        danger_straight = int(self._point_collision(head.rect.x + dir_x * size, head.rect.y + dir_y * size))
        danger_right = int(self._point_collision(head.rect.x + dir_y * size, head.rect.y - dir_x * size))
        danger_left = int(self._point_collision(head.rect.x - dir_y * size, head.rect.y + dir_x * size))

        move_left = int(dir_x == -1)
        move_right = int(dir_x == 1)
        move_up = int(dir_y == -1)
        move_down = int(dir_y == 1)

        food = min(self.foods, key=lambda f: abs(f.rect.x - head.rect.x) + abs(f.rect.y - head.rect.y))
        food_left = int(food.rect.x < head.rect.x)
        food_right = int(food.rect.x > head.rect.x)
        food_up = int(food.rect.y < head.rect.y)
        food_down = int(food.rect.y > head.rect.y)

        return np.array(
            [
                danger_straight,
                danger_right,
                danger_left,
                move_left,
                move_right,
                move_up,
                move_down,
                food_left,
                food_right,
                food_up,
                food_down,
            ],
            dtype=np.float32,
        )

    def head_is_surrounded(self) -> bool:
        
        head_x = self.snake.head.rect.x
        head_y = self.snake.head.rect.y

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for dx, dy in directions:
            if not self._point_collision(head_x + dx, head_y + dy):
                return False
        return True
    
