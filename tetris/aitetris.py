# aitetris.py
import pygame
import random
import numpy as np
from collections import deque, defaultdict
import pickle
import os
import time
import traceback
import argparse
import copy

# --- Constants ---
SCREEN_WIDTH, SCREEN_HEIGHT = 500, 600
PLAYFIELD_WIDTH_PX, PLAYFIELD_HEIGHT_PX = 300, 600 # Area where pieces fall
BLOCK_SIZE = 30
GRID_WIDTH = PLAYFIELD_WIDTH_PX // BLOCK_SIZE # Usually 10 for Tetris
GRID_HEIGHT = PLAYFIELD_HEIGHT_PX // BLOCK_SIZE # Usually 20 for Tetris

INFO_PANEL_WIDTH = SCREEN_WIDTH - PLAYFIELD_WIDTH_PX

# How fast the game runs in 'run' mode (lower is slower)
PLAY_SPEED_FPS = 5 # Frames per second when watching the AI play

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
GRID_COLOR = (40, 40, 40)

# Tetromino Shapes (indices correspond to colors)
TETROMINOES = {
    'I': [(0, 1), (1, 1), (2, 1), (3, 1)],
    'O': [(1, 0), (2, 0), (1, 1), (2, 1)],
    'T': [(0, 1), (1, 1), (2, 1), (1, 0)],
    'S': [(1, 0), (2, 0), (0, 1), (1, 1)],
    'Z': [(0, 0), (1, 0), (1, 1), (2, 1)],
    'J': [(0, 0), (0, 1), (1, 1), (2, 1)],
    'L': [(2, 0), (0, 1), (1, 1), (2, 1)],
}
PIECE_COLORS = [
    (0, 240, 240), # I (Cyan)
    (240, 240, 0), # O (Yellow)
    (160, 0, 240), # T (Purple)
    (0, 240, 0),   # S (Green)
    (240, 0, 0),   # Z (Red)
    (0, 0, 240),   # J (Blue)
    (240, 160, 0)  # L (Orange)
]

# --- Rewards (Optimized Defaults) ---
REWARD_LINE_CLEAR = [0, 100, 300, 500, 800] # Base rewards for 0, 1, 2, 3, 4 lines
REWARD_PER_PIECE_PLACED = 0.1 # Small survival reward
REWARD_GAME_OVER = -1000      # Significant penalty for losing

# Heuristic Penalties (Increased magnitude)
PENALTY_PER_HOLE = -8        # Increased penalty for each hole
PENALTY_AGG_HEIGHT = -4      # Increased penalty based on sum of column heights
PENALTY_BUMPINESS = -2       # Increased penalty for height differences
PENALTY_MAX_HEIGHT = -1      # Added: Penalty based on the highest column
PENALTY_PLACEMENT_HEIGHT_FACTOR = -0.1 # Added: Penalty factor for how high a piece lands

# --- Helper Functions ---
def rotate_point(cx, cy, angle_rad, point):
    """Rotates a point around a center (cx, cy)."""
    x, y = point
    temp_x = x - cx
    temp_y = y - cy
    rotated_x = temp_x * np.cos(angle_rad) - temp_y * np.sin(angle_rad)
    rotated_y = temp_x * np.sin(angle_rad) + temp_y * np.cos(angle_rad)
    return round(rotated_x + cx), round(rotated_y + cy)

# --- Game Classes ---
class Piece:
    def __init__(self, shape_name, start_x, start_y):
        self.shape_name = shape_name
        self.shape = TETROMINOES[shape_name]
        self.color_index = list(TETROMINOES.keys()).index(shape_name)
        self.color = PIECE_COLORS[self.color_index]
        self.x = start_x # Grid coordinates
        self.y = start_y # Grid coordinates
        self.rotation = 0 # 0=0, 1=90, 2=180, 3=270 degrees clockwise

    def get_block_positions(self, x_offset=0, y_offset=0, rotation_override=None):
        """Returns the absolute grid positions of the piece's blocks."""
        positions = []
        current_rotation = rotation_override if rotation_override is not None else self.rotation
        angle_rad = np.radians(current_rotation * 90)
        cx, cy = (1.5, 1.5)
        if self.shape_name == 'I': cx, cy = (1.5, 1.5)
        elif self.shape_name == 'O': cx, cy = (1.5, 0.5)

        for block in self.shape:
            bx, by = block
            if self.shape_name != 'O':
                rotated_bx, rotated_by = rotate_point(cx, cy, angle_rad, (bx, by))
            else:
                rotated_bx, rotated_by = bx, by
            positions.append((self.x + rotated_bx + x_offset, self.y + rotated_by + y_offset))
        return positions

    def rotate(self, clockwise=True):
        """Updates the piece's rotation state."""
        if self.shape_name == 'O': return
        if clockwise:
            self.rotation = (self.rotation + 1) % 4
        else:
            self.rotation = (self.rotation - 1 + 4) % 4

    def move(self, dx, dy):
        """Updates the piece's position."""
        self.x += dx
        self.y += dy

class TetrisGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.well = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.current_piece = None
        self.next_piece_shape = random.choice(list(TETROMINOES.keys()))
        self._new_piece()

    def _new_piece(self):
        """Creates a new falling piece."""
        if self.game_over: return

        self.current_piece = Piece(self.next_piece_shape,
                                   start_x=GRID_WIDTH // 2 - 2,
                                   start_y=0)
        self.next_piece_shape = random.choice(list(TETROMINOES.keys()))

        if not self._is_valid_position(self.current_piece):
            self.game_over = True
            self.current_piece = None

    def _is_valid_position(self, piece, x_offset=0, y_offset=0, rotation_override=None):
        """Checks if piece placement is valid."""
        if piece is None: return False

        block_positions = piece.get_block_positions(x_offset, y_offset, rotation_override)

        for x, y in block_positions:
            if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
                return False
            if y < 0: continue
            if self.well[y][x] != 0:
                return False
        return True

    def _place_piece(self):
        """'Freezes' the current piece onto the well grid."""
        if self.current_piece is None: return

        block_positions = self.current_piece.get_block_positions()
        color_val = self.current_piece.color_index + 1
        for x, y in block_positions:
            if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
                self.well[y][x] = color_val
            elif y < 0:
                 self.game_over = True

        if self.game_over:
             self.current_piece = None

    def _clear_lines(self):
        """Checks for and clears completed lines, returns number cleared."""
        lines_to_clear = []
        for y in range(GRID_HEIGHT):
            if all(self.well[y][x] != 0 for x in range(GRID_WIDTH)):
                lines_to_clear.append(y)

        num_cleared = len(lines_to_clear)
        if num_cleared > 0:
            self.lines_cleared += num_cleared
            # Use the reward structure directly for score update
            self.score += REWARD_LINE_CLEAR[num_cleared]

            lines_to_clear.sort(reverse=True)
            for y_clear in lines_to_clear:
                del self.well[y_clear]
                self.well.insert(0, [0 for _ in range(GRID_WIDTH)])

        return num_cleared

    # --- Removed manual movement/rotation attempts ---
    # AI will directly choose and execute final placement via hard_drop

    def hard_drop_piece(self):
        """Instantly drops the current piece to the lowest valid position and handles placement."""
        lines_cleared = 0
        landed = False
        if self.current_piece and not self.game_over:
            lowest_y = self.current_piece.y
            while self._is_valid_position(self.current_piece, y_offset=(lowest_y - self.current_piece.y + 1)):
                lowest_y += 1

            dy = lowest_y - self.current_piece.y
            # Optional: Add hard drop score bonus (separate from reward shaping for AI)
            # if dy > 0: self.score += dy # Small score bonus per row dropped instantly

            self.current_piece.move(0, dy)

            # Place piece, clear lines, get new piece
            self._place_piece()
            if self.game_over: return 0, True # Game over after placement
            lines_cleared = self._clear_lines()
            self._new_piece() # Get the next piece (also checks for game over)
            landed = True

        return lines_cleared, landed or self.game_over

    # --- Removed step() as AI uses hard_drop ---

    def get_board_features(self, well_state=None):
        """Calculates features for the board state."""
        target_well = well_state if well_state is not None else self.well
        heights = [0] * GRID_WIDTH
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if target_well[y][x] != 0:
                    heights[x] = GRID_HEIGHT - y
                    break
        agg_height = sum(heights)
        holes = 0
        for x in range(GRID_WIDTH):
            block_found = False
            for y in range(GRID_HEIGHT):
                if target_well[y][x] != 0:
                    block_found = True
                elif block_found and target_well[y][x] == 0:
                    holes += 1
        bumpiness = 0
        for x in range(GRID_WIDTH - 1):
            bumpiness += abs(heights[x] - heights[x+1])
        max_height = max(heights) if heights else 0

        # Return tuple: heights(10) + agg_height(1) + holes(1) + bumpiness(1) + max_height(1) = 14 features
        return tuple(heights) + (agg_height, holes, bumpiness, max_height)

    def draw(self, surface, font):
        """Draws the game state."""
        surface.fill(BLACK)

        # Draw Playfield
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                color_val = self.well[y][x]
                rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                if color_val == 0:
                    pygame.draw.rect(surface, BLACK, rect)
                    pygame.draw.rect(surface, GRID_COLOR, rect, 1)
                else:
                    pygame.draw.rect(surface, PIECE_COLORS[color_val - 1], rect)
                    pygame.draw.rect(surface, BLACK, rect, 1)

        # Draw Current Piece
        if self.current_piece and not self.game_over:
            block_positions = self.current_piece.get_block_positions()
            for x, y in block_positions:
                if y >= 0:
                   rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                   pygame.draw.rect(surface, self.current_piece.color, rect)
                   pygame.draw.rect(surface, WHITE, rect, 1)

        # Draw Info Panel
        info_x_start = PLAYFIELD_WIDTH_PX + 10
        if font:
            score_text = font.render(f"Score: {self.score}", True, WHITE)
            surface.blit(score_text, (info_x_start, 10))
            lines_text = font.render(f"Lines: {self.lines_cleared}", True, WHITE)
            surface.blit(lines_text, (info_x_start, 40))
            next_label = font.render("Next:", True, WHITE)
            surface.blit(next_label, (info_x_start, 80))
            if self.next_piece_shape:
                next_p = Piece(self.next_piece_shape, 0, 0)
                next_blocks = next_p.get_block_positions()
                min_x = min(b[0] for b in next_blocks); min_y = min(b[1] for b in next_blocks)
                max_w = max(b[0] for b in next_blocks) - min_x + 1
                # Center the next piece preview horizontally
                preview_base_x = info_x_start + (INFO_PANEL_WIDTH - max_w*BLOCK_SIZE) // 2 - min_x*BLOCK_SIZE
                for x, y in next_blocks:
                    draw_x = preview_base_x + (x) * BLOCK_SIZE # Use absolute block x directly
                    draw_y = 110 + (y - min_y) * BLOCK_SIZE
                    rect = pygame.Rect(draw_x, draw_y, BLOCK_SIZE, BLOCK_SIZE)
                    pygame.draw.rect(surface, next_p.color, rect)
                    pygame.draw.rect(surface, BLACK, rect, 1)

        # Draw Game Over
        if self.game_over:
             try:
                font_large = pygame.font.SysFont(None, 72)
                over_text = font_large.render("GAME OVER", True, (255, 0, 0))
                text_rect = over_text.get_rect(center=(PLAYFIELD_WIDTH_PX // 2, PLAYFIELD_HEIGHT_PX // 2))
                surface.blit(over_text, text_rect)
             except Exception: pass # Ignore font errors here

class QLearningAgent:
    # --- Helper Class within Agent for simulation ---
    class MockGame:
         def __init__(self, well): self.well = well
         def _is_valid_position(self, piece, x_offset=0, y_offset=0, rotation_override=None):
             block_positions = piece.get_block_positions(x_offset, y_offset, rotation_override)
             for x, y in block_positions:
                 if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT): return False
                 if y < 0: continue
                 if self.well[y][x] != 0: return False
             return True
         # Add get_board_features to MockGame for convenience in learn()
         def get_board_features(self, well_state_arg):
              # Use the main game's feature calculation method (static-like call)
              # Requires access to an instance or making the method static - let's pass an instance
              # Or simply call the global function if we refactor get_board_features
              # Simpler: Create a temporary TetrisGame instance here just for the call
              temp_game = TetrisGame()
              features = temp_game.get_board_features(well_state_arg)
              temp_game = None
              return features


    # --- Updated Defaults ---
    def __init__(self, learning_rate=0.02, discount_factor=0.98, exploration_rate=1.0,
                 exploration_decay=0.99995, min_exploration_rate=0.05, q_table_file='tetris.pkl'):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.initial_epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration_rate
        self.q_table_file = q_table_file

        # State features: heights(10) + agg_h(1) + holes(1) + bump(1) + max_h(1) = 14
        self.state_value_table = defaultdict(lambda: 0.0) # Use V(s) approach

        # Determine state length dynamically based on current feature func
        dummy_game = TetrisGame()
        self.current_state_len = len(dummy_game.get_board_features())
        print(f"State representation length: {self.current_state_len}")
        dummy_game = None # Release dummy game

        # Load saved table (passing self.current_state_len for check)
        self.state_value_table = self.load_value_table(self.current_state_len)

    def load_value_table(self, expected_len):
        """Loads the state value table V(s)."""
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'rb') as f:
                    saved_data = pickle.load(f)
                if isinstance(saved_data, dict) and 'value_table' in saved_data and 'epsilon' in saved_data:
                    value_table_loaded = saved_data['value_table']
                    loaded_epsilon = saved_data['epsilon']
                    print(f"Loading agent state from '{self.q_table_file}'...")
                    self.epsilon = max(self.min_epsilon, min(1.0, loaded_epsilon))
                    print(f" - Loaded Epsilon: {self.epsilon:.4f}")

                    value_table = defaultdict(lambda: 0.0)
                    valid_states = 0; invalid_states = 0
                    first_state = next(iter(value_table_loaded), None)
                    loaded_len = len(first_state) if isinstance(first_state, tuple) else -1

                    if first_state is not None and loaded_len != expected_len:
                         print(f"!! State length mismatch! Loaded: {loaded_len}, Expected: {expected_len}. Discarding. !!")
                         self.epsilon = self.initial_epsilon
                         return defaultdict(lambda: 0.0)

                    print(f" - State length matches ({expected_len}). Loading values.")
                    for state, value in value_table_loaded.items():
                        if isinstance(state, tuple) and len(state) == expected_len and isinstance(value, (float, int, np.number)):
                            value_table[state] = float(value)
                            valid_states += 1
                        else: invalid_states += 1
                    print(f" - Loaded States: {len(value_table_loaded)}, Kept: {valid_states}, Discarded: {invalid_states}")
                    if valid_states == 0 and len(value_table_loaded) > 0: self.epsilon = self.initial_epsilon
                    return value_table
                else:
                    print(f"Warning: Unknown format in '{self.q_table_file}'. Starting fresh.")
                    self.epsilon = self.initial_epsilon; return defaultdict(lambda: 0.0)
            except Exception as e:
                print(f"Error loading '{self.q_table_file}': {e}. Starting fresh."); self.epsilon = self.initial_epsilon; return defaultdict(lambda: 0.0)
        else:
            print(f"Agent file '{self.q_table_file}' not found. Starting fresh."); self.epsilon = self.initial_epsilon; return defaultdict(lambda: 0.0)

    def save_q_table(self):
        """Saves the state value table V(s) and epsilon."""
        try:
            data_to_save = {'value_table': dict(self.state_value_table), 'epsilon': self.epsilon}
            with open(self.q_table_file, 'wb') as f: pickle.dump(data_to_save, f)
        except Exception as e: print(f"Error saving data to '{self.q_table_file}': {e}")

    def get_state(self, game):
        """Returns the feature tuple for the current game state."""
        return game.get_board_features()

    def _calculate_placement_outcome(self, game, piece, rotation, x_pos):
        """Simulates placing the piece, returns (resulting_well, lines_cleared, resulting_features, is_valid, landing_y)."""
        if piece is None: return None, 0, None, False, -1

        hypothetical_piece = copy.deepcopy(piece)
        hypothetical_piece.rotation = rotation
        hypothetical_piece.x = x_pos
        hypothetical_piece.y = 0 # Start high

        # Ensure valid start, adjusting y if needed (for I piece etc.)
        initial_y = 0
        while not game._is_valid_position(hypothetical_piece, y_offset=initial_y - hypothetical_piece.y) and initial_y > -3:
             initial_y -= 1
        hypothetical_piece.y = initial_y
        if not game._is_valid_position(hypothetical_piece): return None, 0, None, False, -1 # Invalid spawn

        # Simulate hard drop
        landing_y = hypothetical_piece.y
        while game._is_valid_position(hypothetical_piece, y_offset=(landing_y - hypothetical_piece.y + 1)):
            landing_y += 1
        hypothetical_piece.y = landing_y

        if not game._is_valid_position(hypothetical_piece): return None, 0, None, False, landing_y

        # Simulate placing on a copy of the well
        temp_well = [row[:] for row in game.well]
        block_positions = hypothetical_piece.get_block_positions()
        color_val = hypothetical_piece.color_index + 1
        placed_above_screen = False
        min_block_y = GRID_HEIGHT # Track highest placed block y (lowest value)

        for x, y in block_positions:
            if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
                temp_well[y][x] = color_val
                min_block_y = min(min_block_y, y)
            elif y < 0: placed_above_screen = True; break

        if placed_above_screen:
            features = game.get_board_features(temp_well) # Features might be weird
            return temp_well, 0, features, True, landing_y # Valid placement, but bad outcome

        # Simulate clearing lines
        lines_cleared = 0
        rows_to_clear = []
        for y in range(GRID_HEIGHT):
            if all(temp_well[y][x] != 0 for x in range(GRID_WIDTH)): rows_to_clear.append(y)
        lines_cleared = len(rows_to_clear)
        if lines_cleared > 0:
            rows_to_clear.sort(reverse=True)
            for y_clear in rows_to_clear: del temp_well[y_clear]; temp_well.insert(0, [0]*GRID_WIDTH)

        resulting_features = game.get_board_features(temp_well)
        return temp_well, lines_cleared, resulting_features, True, landing_y


    def choose_action(self, game):
        """Evaluates placements, returns best (target_rotation, target_x_pos)."""
        if game.current_piece is None or game.game_over: return None

        possible_placements = [] # (rotation, x_pos, desirability, features_after)
        current_piece = game.current_piece

        for rot in range(4):
            test_piece = Piece(current_piece.shape_name, 0, 0)
            blocks = test_piece.get_block_positions(rotation_override=rot)
            if not blocks: continue # Should not happen
            min_block_x = min(b[0] for b in blocks); max_block_x = max(b[0] for b in blocks)

            for x_pos in range(0 - min_block_x, GRID_WIDTH - max_block_x):
                well_after, lines_cleared, features_after, is_valid, landing_y = \
                    self._calculate_placement_outcome(game, current_piece, rot, x_pos)

                if is_valid:
                    immediate_reward = REWARD_LINE_CLEAR[lines_cleared] + REWARD_PER_PIECE_PLACED

                    # Heuristic penalties based on features_after
                    # features_after = (heights..., agg_h, holes, bumpiness, max_h) - len=14
                    agg_height = features_after[GRID_WIDTH]
                    holes = features_after[GRID_WIDTH + 1]
                    bumpiness = features_after[GRID_WIDTH + 2]
                    max_height = features_after[GRID_WIDTH + 3]

                    # Placement height penalty (based on where the piece lands)
                    placement_height_penalty = (GRID_HEIGHT - landing_y) * abs(PENALTY_PLACEMENT_HEIGHT_FACTOR)

                    heuristic_penalty = (holes * abs(PENALTY_PER_HOLE) +
                                        agg_height * abs(PENALTY_AGG_HEIGHT) +
                                        bumpiness * abs(PENALTY_BUMPINESS) +
                                        max_height * abs(PENALTY_MAX_HEIGHT) +
                                        placement_height_penalty)

                    game_over_penalty = 0
                    if any(well_after[0][x] != 0 for x in range(GRID_WIDTH)):
                         game_over_penalty = abs(REWARD_GAME_OVER) * 1.5 # Heavy penalty

                    estimated_future_value = self.state_value_table[features_after]

                    desirability = (immediate_reward + self.gamma * estimated_future_value -
                                    heuristic_penalty - game_over_penalty)

                    possible_placements.append((rot, x_pos, desirability, features_after))

        if not possible_placements: return (current_piece.rotation, current_piece.x) # Fallback

        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            chosen_placement_data = random.choice(possible_placements)
        else:
            possible_placements.sort(key=lambda x: x[2], reverse=True)
            chosen_placement_data = possible_placements[0]

        return (chosen_placement_data[0], chosen_placement_data[1])

    def learn(self, state_features_before, reward_received, state_features_after, next_piece_info):
        """Update V(s_before) using TD learning."""
        current_value = self.state_value_table[state_features_before]

        # Find max V(s'') achievable from state_features_after with next_piece
        max_future_value = 0.0
        if next_piece_info is not None:
            next_piece_shape = next_piece_info['shape']
            well_state_after = next_piece_info['well'] # Well corresponding to state_features_after
            mock_game = self.MockGame(well_state_after) # Use helper class instance
            next_piece_obj = Piece(next_piece_shape, 0, 0)

            best_v_s_prime = -float('inf')
            found_valid_next_placement = False

            for rot in range(4):
                test_piece = Piece(next_piece_shape, 0, 0)
                blocks = test_piece.get_block_positions(rotation_override=rot)
                if not blocks: continue
                min_bx = min(b[0] for b in blocks); max_bx = max(b[0] for b in blocks)

                for x_pos in range(0 - min_bx, GRID_WIDTH - max_bx):
                    # Simulate outcome of placing the *next* piece using the mock game
                    # Need to use _calculate_placement_outcome with the mock_game context
                    # We only need features_s_prime and validity from this call
                    _well_sp, _lines_sp, features_s_prime, is_valid_sp, _ly_sp = \
                        self._calculate_placement_outcome(mock_game, next_piece_obj, rot, x_pos)

                    if is_valid_sp:
                        found_valid_next_placement = True
                        best_v_s_prime = max(best_v_s_prime, self.state_value_table[features_s_prime])

            # If no valid placements were found for the next piece (e.g., instant game over), future value is 0
            if found_valid_next_placement:
                 max_future_value = best_v_s_prime if best_v_s_prime > -float('inf') else 0.0

        # TD Update: V(s) = V(s) + alpha * (Reward + gamma * max V(s') - V(s))
        td_target = reward_received + self.gamma * max_future_value
        new_value = current_value + self.lr * (td_target - current_value)
        self.state_value_table[state_features_before] = new_value

    def decay_exploration(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)


# --- Game Loop Function ---
def run_game(agent, train=False, num_episodes=1000, render=False, max_steps_per_episode=5000, auto_reset_config=None):
    pygame.init()
    screen = None; font = None; clock = None

    if render:
        try:
            pygame.font.init(); font = pygame.font.SysFont(None, 30)
        except Exception as e: print(f"Warning: Pygame font failed: {e}. Text disabled."); font = None
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Q-Learning Tetris AI')
        clock = pygame.time.Clock()

    session_high_score = 0; scores = []; total_lines_session = 0; total_pieces_session_log = 0
    mode = "Training" if train else "Playing"; render_status = "Visible" if render else "Non-Visible"
    print(f"\n--- Starting {mode} ({render_status}) ---"); print(f"Start Epsilon: {agent.epsilon:.4f}")
    if train: print(f"Episodes: {num_episodes}, Max Steps/Ep: {max_steps_per_episode}")
    else: print("Using learned policy (minimal exploration)."); agent.epsilon = agent.min_epsilon

    # Auto-reset setup
    best_avg_score_so_far = -np.inf; stagnation_counter = 0; log_interval = 50
    auto_reset_enabled = train and auto_reset_config is not None
    if auto_reset_enabled: print(f"Auto Epsilon Reset: Patience={auto_reset_config['patience']}, Value={auto_reset_config['value']:.2f}")

    for episode in range(num_episodes):
        game = TetrisGame(); steps_taken = 0; episode_over = False; last_state_features = None

        while not episode_over and steps_taken < max_steps_per_episode:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nQuit signal.");
                        if train: agent.save_q_table(); print("Agent saved.")
                        pygame.quit(); return scores, session_high_score
                # No manual controls needed for AI play

            if game.game_over:
                episode_over = True
                reward_for_learning = REWARD_GAME_OVER # Final penalty
                if train and last_state_features is not None:
                    # Learn from the state leading to game over (future value is 0)
                    td_target = reward_for_learning
                    current_value = agent.state_value_table[last_state_features]
                    new_value = current_value + agent.lr * (td_target - current_value)
                    agent.state_value_table[last_state_features] = new_value
                continue

            # --- AI Decision & Execution ---
            if game.current_piece:
                state_features_before = agent.get_state(game)
                last_state_features = state_features_before # Remember for learning

                chosen_placement = agent.choose_action(game)

                if chosen_placement:
                    target_rotation, target_x = chosen_placement
                    original_piece = copy.deepcopy(game.current_piece) # Keep ref before drop

                    # Simulate *before* executing to get expected features_after and landing_y
                    well_after_sim, lines_sim, features_after_sim, is_valid_sim, landing_y_sim = \
                         agent._calculate_placement_outcome(game, original_piece, target_rotation, target_x)

                    # Execute the placement in the real game
                    game.current_piece.rotation = target_rotation
                    game.current_piece.x = target_x
                    lines_cleared_actual, landed = game.hard_drop_piece() # Executes placement

                    # Determine reward based on actual outcome + simulated penalties
                    reward_for_learning = 0
                    if game.game_over: # Check if hard_drop caused game over
                        reward_for_learning = REWARD_GAME_OVER
                        episode_over = True
                    elif is_valid_sim: # If simulation was valid (should be if placement chosen)
                        reward_for_learning = REWARD_LINE_CLEAR[lines_cleared_actual] + REWARD_PER_PIECE_PLACED

                        # Calculate penalties based on the simulated outcome state (features_after_sim)
                        agg_h_sim = features_after_sim[GRID_WIDTH]
                        holes_sim = features_after_sim[GRID_WIDTH + 1]
                        bump_sim = features_after_sim[GRID_WIDTH + 2]
                        max_h_sim = features_after_sim[GRID_WIDTH + 3]
                        placement_penalty_sim = (GRID_HEIGHT - landing_y_sim) * abs(PENALTY_PLACEMENT_HEIGHT_FACTOR)

                        reward_for_learning += PENALTY_AGG_HEIGHT * agg_h_sim
                        reward_for_learning += PENALTY_PER_HOLE * holes_sim
                        reward_for_learning += PENALTY_BUMPINESS * bump_sim
                        reward_for_learning += PENALTY_MAX_HEIGHT * max_h_sim
                        reward_for_learning += placement_penalty_sim
                    else:
                         # Should not happen if choose_action returns valid placement
                         print("Warning: Executed placement based on invalid simulation?")
                         reward_for_learning = REWARD_GAME_OVER # Penalize heavily

                    # Learn
                    if train:
                        next_piece_info = None
                        if not game.game_over:
                             next_piece_info = {'shape': game.next_piece_shape, 'well': game.well}
                        # Use simulated features after for learning consistency
                        agent.learn(state_features_before, reward_for_learning, features_after_sim, next_piece_info)

                    steps_taken += 1
                    if train: total_pieces_session_log += 1
                else:
                     # No valid placement found - should be rare, but maybe force game over
                     print("Warning: No valid placement chosen by AI.")
                     game.game_over = True # Treat as game over if AI cannot move
                     episode_over = True

            else: # Should not happen if game not over
                 if not game.game_over: print("Warning: No current piece but game not over?"); time.sleep(0.1)


            if render:
                screen.fill(BLACK); game.draw(screen, font)
                if font:
                     ep_text = font.render(f"Ep: {episode+1}/{num_episodes}", True, WHITE)
                     eps_text = font.render(f"Eps: {agent.epsilon:.3f}", True, WHITE)
                     screen.blit(ep_text, (PLAYFIELD_WIDTH_PX + 10, SCREEN_HEIGHT - 60))
                     screen.blit(eps_text, (PLAYFIELD_WIDTH_PX + 10, SCREEN_HEIGHT - 30))
                pygame.display.flip()

                # --- Speed Control ---
                if clock:
                    if train: clock.tick(60) # Train with rendering runs faster
                    else: clock.tick(PLAY_SPEED_FPS) # Run mode uses slower FPS

        # --- Episode End ---
        if game.score > session_high_score: session_high_score = game.score
        scores.append(game.score); total_lines_session += game.lines_cleared

        if train:
            agent.decay_exploration()
            if (episode + 1) % log_interval == 0:
                avg_score = np.mean(scores[-log_interval:]) if scores else 0
                avg_lines = total_lines_session / log_interval if log_interval > 0 else 0
                avg_pieces = total_pieces_session_log / log_interval if log_interval > 0 else 0
                print_msg = f"Ep {episode + 1}/{num_episodes} | AvgScore({log_interval}): {avg_score:.0f} | AvgLines: {avg_lines:.1f} | AvgPieces: {avg_pieces:.1f} | High: {session_high_score} | Eps: {agent.epsilon:.4f} | States: {len(agent.state_value_table)}"

                if auto_reset_enabled:
                    if best_avg_score_so_far == -np.inf and (episode + 1) >= log_interval: best_avg_score_so_far = avg_score; print_msg += " | Init best avg."
                    elif (episode + 1) > log_interval:
                        improvement = avg_score - best_avg_score_so_far
                        if improvement > 1.0: # Require minimal improvement
                            print_msg += f" | New Best Avg! (+{improvement:.0f})"; best_avg_score_so_far = avg_score; stagnation_counter = 0
                        else:
                            stagnation_counter += 1; print_msg += f" | Stagnated ({stagnation_counter}/{auto_reset_config['patience']})"
                            if stagnation_counter >= auto_reset_config['patience']:
                                old_epsilon = agent.epsilon; agent.epsilon = max(agent.min_epsilon, auto_reset_config['value'])
                                print_msg += f" | !!! AUTO-RESET Eps ({old_epsilon:.4f} -> {agent.epsilon:.4f}) !!!"; stagnation_counter = 0; best_avg_score_so_far = avg_score
                print(print_msg); agent.save_q_table(); total_lines_session = 0; total_pieces_session_log = 0

        if not train: print(f"Game {episode+1} finished! Score: {game.score}, Lines: {game.lines_cleared}, Pieces: {steps_taken}")
        if not train and render: time.sleep(1) # Pause briefly between games in run mode

    # --- End of Run ---
    if train: print("\n--- Training Finished ---"); agent.save_q_table(); print(f"Final agent state saved.")
    if render: pygame.quit()
    return scores, session_high_score


# --- Main Execution Block ---
if __name__ == '__main__':
    DEFAULT_RESET_PATIENCE = 10 # Default patience for auto-reset
    DEFAULT_RESET_VALUE = 0.6   # Default reset epsilon value

    parser = argparse.ArgumentParser(description="Train or Run a Q-Learning Tetris AI.")
    parser.add_argument('mode', metavar='MODE', type=str, nargs='?', choices=['train', 'run'], default='run', help="Operation mode: 'train' or 'run'.")
    # --- Adjusted default episodes ---
    parser.add_argument('-e', '--episodes', type=int, default=10000, help="Episodes for training (default: 10000).")
    parser.add_argument('-p', '--play-episodes', type=int, default=5, help="Episodes to watch during 'run'.")
    parser.add_argument('--render-train', action='store_true', help="Render during training (SLOW).")
    parser.add_argument('--fresh', action='store_true', help="Force fresh training.")
    parser.add_argument('--qtable', type=str, default="tetris.pkl", help="Agent state filename.")
    parser.add_argument('--max-steps', type=int, default=3000, help="Max pieces per episode.")
    parser.add_argument('--reset-epsilon', type=float, default=None, help="Manually set starting epsilon.")
    parser.add_argument('--auto-reset', action='store_true', help="Enable auto epsilon reset during training.")
    parser.add_argument('--reset-patience', type=int, default=DEFAULT_RESET_PATIENCE, help=f"Auto-Reset patience intervals (default: {DEFAULT_RESET_PATIENCE}).")
    parser.add_argument('--reset-value', type=float, default=DEFAULT_RESET_VALUE, help=f"Auto-Reset epsilon value (default: {DEFAULT_RESET_VALUE}).")

    args = parser.parse_args()

    DO_TRAINING = (args.mode == 'train')
    RENDER_GAME = (not DO_TRAINING) or args.render_train # Render if 'run' or if 'train --render-train'
    NUM_EPISODES_ARG = args.episodes if DO_TRAINING else args.play_episodes
    AGENT_STATE_FILENAME = args.qtable
    MAX_STEPS = args.max_steps
    LOAD_AGENT_STATE = not args.fresh

    auto_reset_params = None
    if DO_TRAINING and args.auto_reset:
        auto_reset_params = {'patience': args.reset_patience, 'value': max(0.01, min(1.0, args.reset_value))}

    if args.mode == 'run' and not os.path.exists(AGENT_STATE_FILENAME):
        print(f"Error: Cannot run. Agent state file '{AGENT_STATE_FILENAME}' not found."); exit(1)

    if DO_TRAINING and LOAD_AGENT_STATE and os.path.exists(AGENT_STATE_FILENAME): print(f"Found '{AGENT_STATE_FILENAME}'. Checking compatibility...")
    elif DO_TRAINING and LOAD_AGENT_STATE and not os.path.exists(AGENT_STATE_FILENAME): print(f"Warning: Load requested but '{AGENT_STATE_FILENAME}' not found. Starting fresh."); LOAD_AGENT_STATE = False
    elif DO_TRAINING and not LOAD_AGENT_STATE: print(f"Starting fresh training (--fresh specified).")

    print("-" * 30)
    # Agent creation uses the new defaults defined in the QLearningAgent class
    agent = QLearningAgent(q_table_file=AGENT_STATE_FILENAME)
    print("-" * 30)

    if args.reset_epsilon is not None:
         manual_eps_val = max(agent.min_epsilon, min(1.0, args.reset_epsilon))
         print(f"!!! MANUALLY Setting starting Epsilon to: {manual_eps_val:.4f} !!!")
         agent.epsilon = manual_eps_val; agent.initial_epsilon = manual_eps_val

    session_start_time = time.time(); session_high_score_final = 0; session_scores = []; e_type = None

    try:
        session_scores, session_high_score_final = run_game(
            agent, train=DO_TRAINING, num_episodes=NUM_EPISODES_ARG, render=RENDER_GAME,
            max_steps_per_episode=MAX_STEPS, auto_reset_config=auto_reset_params
        )
    except KeyboardInterrupt: print(f"\n{args.mode.capitalize()} interrupted."); e_type = KeyboardInterrupt
    except Exception as e: print(f"\nUnexpected error: {e}"); traceback.print_exc(); e_type = type(e)
    finally:
        if DO_TRAINING and (e_type is not None or args.mode == 'train'):
             print("Attempting to save agent state..."); agent.save_q_table()
             if os.path.exists(agent.q_table_file): print(f"Agent state saved to '{agent.q_table_file}'.")
             else: print(f"Warning: Failed to save agent state.")

        session_end_time = time.time()
        print(f"\n--- {args.mode.capitalize()} Session Summary ---")
        print(f"Duration: {session_end_time - session_start_time:.2f} seconds.")
        if session_scores:
             print(f"Episodes completed: {len(session_scores)}")
             print(f"Average score: {np.mean(session_scores):.2f}")
             print(f"Highest score: {session_high_score_final}")
        else: print("No full episodes completed or interrupted early.")
        if DO_TRAINING: print(f"Final Epsilon: {agent.epsilon:.4f}"); print(f"Total unique states learned: {len(agent.state_value_table)}")

    print("\nScript finished.")