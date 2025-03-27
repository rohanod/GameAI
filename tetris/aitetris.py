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
import copy # Needed for deep copies of board states

# --- Game Constants ---
WIDTH, HEIGHT = 300, 600
GRID_WIDTH, GRID_HEIGHT = 10, 20
BLOCK_SIZE = HEIGHT // GRID_HEIGHT
TOP_MARGIN = 50
WINDOW_WIDTH = GRID_WIDTH * BLOCK_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * BLOCK_SIZE + TOP_MARGIN

INITIAL_FALL_DELAY = 500; FAST_FALL_DELAY = 50; INPUT_DELAY = 100

WHITE = (255, 255, 255); BLACK = (0, 0, 0); GRAY = (128, 128, 128)
RED = (213, 50, 80); GREEN = (0, 200, 0); BLUE = (50, 80, 213)
CYAN = (0, 255, 255); MAGENTA = (255, 0, 255); YELLOW = (255, 255, 0); ORANGE = (255, 165, 0)
PIECE_COLORS = [GRAY, CYAN, YELLOW, MAGENTA, GREEN, RED, BLUE, ORANGE]

SHAPES = [
    [[[0,0], [1,0], [2,0], [3,0]], [[0,0], [0,1], [0,2], [0,3]]], # I
    [[[0,0], [0,1], [1,0], [1,1]]], # O
    [[[0,1], [1,0], [1,1], [1,2]], [[0,1], [1,1], [1,0], [2,1]], [[1,0], [1,1], [1,2], [2,1]], [[0,0], [1,0], [1,1], [2,0]]], # T
    [[[0,1], [0,2], [1,0], [1,1]], [[0,0], [1,0], [1,1], [2,1]]], # S
    [[[0,0], [0,1], [1,1], [1,2]], [[0,2], [1,1], [1,2], [2,1]]], # Z
    [[[0,0], [1,0], [1,1], [1,2]], [[0,1], [0,2], [1,1], [2,1]], [[1,0], [1,1], [1,2], [2,2]], [[0,1], [1,1], [2,0], [2,1]]], # J
    [[[0,2], [1,0], [1,1], [1,2]], [[0,1], [1,1], [2,1], [2,2]], [[1,0], [1,1], [1,2], [2,0]], [[0,0], [0,1], [1,1], [2,1]]]  # L
]

# --- AI Constants --- (Strong Penalties + No Clear Streak + Eroded Cells)
REWARD_LINE_CLEAR = [0, 100, 300, 700, 1500] # AI Reward for line clears
REWARD_GAME_OVER = -1000

REWARD_PER_PIECE = 0
REWARD_STEP_PENALTY = -0.01

# Drastically Increased Penalties applied when a piece locks
REWARD_HOLE_PENALTY = -75        # VERY Strong hole penalty
REWARD_BUMPINESS_PENALTY = -3    # Increased bumpiness penalty
REWARD_AGG_HEIGHT_PENALTY = -5.0 # VERY Strong aggregate height penalty
REWARD_MAX_HEIGHT_PENALTY = -10  # VERY Strong max height penalty

# Penalty for placing pieces without clearing lines
NO_CLEAR_STREAK_THRESHOLD = 10 # Apply penalty after this many consecutive non-clearing locks
REWARD_NO_CLEAR_STREAK_PENALTY = -50 # Penalty value (tune this)

# Reward for efficient line clears
REWARD_ERODED_CELL_MULTIPLIER = 5 # Bonus per block of the placed piece that cleared a line (tune this)

REWARD_ROW_PROGRESS_FACTOR = 0 # Intermediate reward disabled

# Actions
ACTION_LEFT = 0; ACTION_RIGHT = 1; ACTION_ROTATE = 2; ACTION_NONE = 3; NUM_ACTIONS = 4

class Tetromino:
    def __init__(self, shape_index, grid_width):
        self.shape_index = shape_index; self.shapes = SHAPES[shape_index]; self.rotation = 0
        self.color_index = shape_index + 1; self.x = grid_width // 2 - 1; self.y = 0
    def get_blocks(self): return [(self.y + r, self.x + c) for r, c in self.shapes[self.rotation]]
    def rotate(self): self.rotation = (self.rotation + 1) % len(self.shapes)
    def unrotate(self): self.rotation = (self.rotation - 1) % len(self.shapes)
    def move(self, dx, dy): self.x += dx; self.y += dy

class TetrisGame:
    def __init__(self):
        self.grid_width = GRID_WIDTH; self.grid_height = GRID_HEIGHT; self.reset()

    def reset(self):
        self.board = np.zeros((self.grid_height, self.grid_width), dtype=int); self.score = 0
        self.lines_cleared_total = 0; self.game_over = False; self.current_piece = self._new_piece()
        self.next_piece = self._new_piece(); self.fall_time = 0; self.fall_delay = INITIAL_FALL_DELAY
        self.last_input_time = 0
        self.no_clear_streak = 0 # Initialize the streak counter
        return self._get_game_state()

    def _new_piece(self): return Tetromino(random.randint(0, len(SHAPES) - 1), self.grid_width)

    def _is_valid_position(self, piece=None):
        p = piece or self.current_piece; blocks = p.get_blocks()
        for r, c in blocks:
            if not (0 <= c < self.grid_width and 0 <= r < self.grid_height): return False
            if r >= 0 and self.board[r, c] != 0: return False
        return True

    # Modified _clear_lines to return indices
    def _clear_lines(self):
        """Checks for and clears completed lines. Returns number cleared AND indices cleared."""
        lines_to_clear = [r for r in range(self.grid_height) if np.all(self.board[r] != 0)]
        num_cleared = len(lines_to_clear)
        if num_cleared > 0:
            self.board = np.delete(self.board, lines_to_clear, axis=0)
            new_lines = np.zeros((num_cleared, self.grid_width), dtype=int)
            self.board = np.vstack((new_lines, self.board))
        return num_cleared, lines_to_clear # Return indices

    def _lock_piece(self):
        # Store coordinates BEFORE locking
        placed_piece_coords = self.current_piece.get_blocks()

        piece_locked_y = -1
        for r, c in placed_piece_coords: # Use stored coords
            if 0 <= r < self.grid_height and 0 <= c < self.grid_width:
                self.board[r, c] = self.current_piece.color_index
                if r > piece_locked_y: piece_locked_y = r
            else: pass

        # Game Over Check
        spawn_blocked = False; test_piece_spawn = Tetromino(self.next_piece.shape_index, self.grid_width)
        for r_offset, c_offset in test_piece_spawn.shapes[0]:
            sr, sc = test_piece_spawn.y + r_offset, test_piece_spawn.x + c_offset
            if 0 <= sr < self.grid_height and 0 <= sc < self.grid_width and self.board[sr, sc] != 0: spawn_blocked = True; break
            elif sr < 0: pass
        if piece_locked_y < 1 or spawn_blocked: self.game_over = True

        placement_reward = 0; lines_cleared = 0; no_clear_streak_penalty = 0; eroded_reward = 0
        cleared_line_indices = []

        if not self.game_over:
            # Call modified _clear_lines
            lines_cleared, cleared_line_indices = self._clear_lines()

            # Update No Clear Streak Counter & Penalty
            if lines_cleared == 0:
                self.no_clear_streak += 1
                if self.no_clear_streak >= NO_CLEAR_STREAK_THRESHOLD:
                    no_clear_streak_penalty = REWARD_NO_CLEAR_STREAK_PENALTY
            else:
                self.no_clear_streak = 0 # Reset streak

            # Calculate Eroded Piece Cells Reward
            if lines_cleared > 0:
                eroded_cells = 0
                for r, c in placed_piece_coords: # Check original coords
                    if r in cleared_line_indices: # Was this block in a cleared row?
                         if 0 <= r < self.grid_height and 0 <= c < self.grid_width: # Bounds check
                            eroded_cells += 1
                eroded_reward = eroded_cells * REWARD_ERODED_CELL_MULTIPLIER

            # Base Game Score Update
            if lines_cleared == 1: self.score += 100
            elif lines_cleared == 2: self.score += 300
            elif lines_cleared == 3: self.score += 500
            elif lines_cleared >= 4: self.score += 1200
            self.lines_cleared_total += lines_cleared

            # AI Reward Calculation
            board_features = self._calculate_board_features(self.board)
            line_clear_bonus = REWARD_LINE_CLEAR[lines_cleared]
            holes_penalty = board_features['holes'] * REWARD_HOLE_PENALTY
            bumpiness_penalty = board_features['bumpiness'] * REWARD_BUMPINESS_PENALTY
            agg_height_penalty = board_features['aggregate_height'] * REWARD_AGG_HEIGHT_PENALTY
            max_height_penalty = board_features['max_height'] * REWARD_MAX_HEIGHT_PENALTY

            # Combine ALL reward components
            placement_reward = (line_clear_bonus + REWARD_PER_PIECE +
                                holes_penalty + bumpiness_penalty +
                                agg_height_penalty + max_height_penalty +
                                no_clear_streak_penalty +
                                eroded_reward) # Add eroded reward

            self.current_piece = self.next_piece; self.next_piece = self._new_piece()
            if not self._is_valid_position(self.current_piece): self.game_over = True; placement_reward = REWARD_GAME_OVER

        return placement_reward, self.game_over

    def step(self, action):
        if self.game_over: return self._get_game_state(), REWARD_GAME_OVER, self.game_over, self.score

        moved_or_rotated = False
        if action == ACTION_LEFT:
            self.current_piece.move(-1, 0)
            if not self._is_valid_position(): self.current_piece.move(1, 0)
            else: moved_or_rotated = True
        elif action == ACTION_RIGHT:
            self.current_piece.move(1, 0)
            if not self._is_valid_position(): self.current_piece.move(-1, 0)
            else: moved_or_rotated = True
        elif action == ACTION_ROTATE:
            original_rot = self.current_piece.rotation; self.current_piece.rotate()
            if not self._is_valid_position():
                kicks = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1)]
                kicked = False; current_x, current_y = self.current_piece.x, self.current_piece.y
                for dx, dy in kicks:
                    self.current_piece.move(dx, dy)
                    if self._is_valid_position(): kicked = True; moved_or_rotated = True; break
                    else: self.current_piece.move(-dx, -dy)
                if not kicked: self.current_piece.x = current_x; self.current_piece.y = current_y; self.current_piece.rotation = original_rot
            else: moved_or_rotated = True

        self.current_piece.move(0, 1)
        reward = REWARD_STEP_PENALTY

        if not self._is_valid_position():
            self.current_piece.move(0, -1)
            placement_reward, game_over_after_lock = self._lock_piece()
            self.game_over = game_over_after_lock
            reward = REWARD_GAME_OVER if self.game_over else reward + placement_reward

        current_score = self.score
        return self._get_game_state(), reward, self.game_over, current_score

    def _get_game_state(self):
        return {"board": self.board.copy(), "current_piece": copy.deepcopy(self.current_piece), "game_over": self.game_over}

    @staticmethod
    def _calculate_board_features(board):
        height, width = board.shape; heights = np.zeros(width, dtype=int)
        for c in range(width):
            filled = np.where(board[:, c] != 0)[0]; heights[c] = height - np.min(filled) if len(filled) > 0 else 0
        agg_h = np.sum(heights); holes = 0; bump = 0
        for c in range(width):
            h = heights[c]
            if h > 0: holes += np.sum(board[height - h:height, c] == 0)
            if c < width - 1: bump += abs(heights[c] - heights[c+1])
        max_h = np.max(heights) if np.any(heights) else 0
        return {"heights": heights, "aggregate_height": agg_h, "holes": holes, "bumpiness": bump, "max_height": max_h}

    def draw(self, surface, font):
        surface.fill(BLACK)
        for r in range(self.grid_height): pygame.draw.line(surface, GRAY, (0, r*BLOCK_SIZE+TOP_MARGIN), (WINDOW_WIDTH, r*BLOCK_SIZE+TOP_MARGIN), 1)
        for c in range(self.grid_width + 1): pygame.draw.line(surface, GRAY, (c*BLOCK_SIZE, TOP_MARGIN), (c*BLOCK_SIZE, WINDOW_HEIGHT), 1)
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                if self.board[r,c] != 0:
                    color = PIECE_COLORS[self.board[r,c]%len(PIECE_COLORS)]; pygame.draw.rect(surface, color, (c*BLOCK_SIZE+1, r*BLOCK_SIZE+TOP_MARGIN+1, BLOCK_SIZE-2, BLOCK_SIZE-2))
        if self.current_piece and not self.game_over:
            color = PIECE_COLORS[self.current_piece.color_index%len(PIECE_COLORS)]
            for r, c in self.current_piece.get_blocks():
                 if r >= 0: pygame.draw.rect(surface, color, (c*BLOCK_SIZE+1, r*BLOCK_SIZE+TOP_MARGIN+1, BLOCK_SIZE-2, BLOCK_SIZE-2))
        pygame.draw.rect(surface, WHITE, (0, TOP_MARGIN, WINDOW_WIDTH, WINDOW_HEIGHT - TOP_MARGIN), 2)
        if font:
            scr_txt=font.render(f"Score:{self.score}",1,WHITE); lin_txt=font.render(f"Lines:{self.lines_cleared_total}",1,WHITE); surface.blit(scr_txt,(5,5)); surface.blit(lin_txt,(5,25))
        if self.game_over and font:
            ovr_fnt=pygame.font.SysFont(None, 50); ovr_txt=ovr_fnt.render("GAME OVER", 1, RED); ovr_rct=ovr_txt.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2)); surface.blit(ovr_txt,ovr_rct)

# --- Q-Learning Agent ---
class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0,
                 exploration_decay=0.9999, min_exploration_rate=0.01, q_table_file='tetris.pkl'):
        self.lr = learning_rate; self.gamma = discount_factor; self.initial_epsilon = exploration_rate
        self.epsilon = exploration_rate; self.epsilon_decay = exploration_decay; self.min_epsilon = min_exploration_rate
        self.q_table_file = q_table_file; self.actions = list(range(NUM_ACTIONS)); self.num_actions = NUM_ACTIONS
        dummy_game = TetrisGame(); dummy_state_info = dummy_game._get_game_state()
        try: self.current_state_tuple_len = len(self.get_state(dummy_state_info))
        except Exception: self.current_state_tuple_len = 7 # Fallback
        self.q_table = self.load_q_table()

    def load_q_table(self):
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'rb') as f: saved_data = pickle.load(f)
                if isinstance(saved_data, dict) and 'q_table' in saved_data and 'epsilon' in saved_data:
                    q_table_loaded, loaded_epsilon = saved_data['q_table'], saved_data['epsilon']
                    print(f"Loaded '{self.q_table_file}' ({len(q_table_loaded)} states, Eps {loaded_epsilon:.4f}).")
                    self.epsilon = loaded_epsilon
                else: q_table_loaded = saved_data; print(f"Loaded old format '{self.q_table_file}'. Reset eps."); self.epsilon = self.initial_epsilon

                q_table = defaultdict(lambda: np.zeros(self.num_actions)); valid, invalid = 0, 0
                for state, values in q_table_loaded.items():
                    q_vals = np.zeros(self.num_actions)
                    if isinstance(values, (list, np.ndarray)) and len(values) == self.num_actions: q_vals = np.array(values)
                    elif isinstance(values, np.ndarray) and values.shape == (self.num_actions,): q_vals = values
                    if isinstance(state, tuple) and len(state) == self.current_state_tuple_len: q_table[state] = q_vals; valid += 1
                    else: invalid += 1
                if invalid > 0: print(f"Warn: Discarded {invalid} states (fmt mismatch). Kept {valid}.")
                return q_table
            except Exception as e: print(f"Error load '{self.q_table_file}': {e}. Start fresh."); self.epsilon = self.initial_epsilon; return defaultdict(lambda: np.zeros(self.num_actions))
        else: print(f"'{self.q_table_file}' not found. Start fresh."); self.epsilon = self.initial_epsilon; return defaultdict(lambda: np.zeros(self.num_actions))

    def save_q_table(self):
        try:
            with open(self.q_table_file, 'wb') as f: pickle.dump({'q_table': dict(self.q_table), 'epsilon': self.epsilon}, f)
        except Exception as e: print(f"Error save '{self.q_table_file}': {e}")

    def get_state(self, game_state_info):
        board = game_state_info["board"]; piece = game_state_info["current_piece"]
        if piece is None: return (0,) * self.current_state_tuple_len
        features = TetrisGame._calculate_board_features(board)
        state = (int(features['aggregate_height'] // 15), int(features['holes']), int(features['bumpiness'] // 4), int(features['max_height'] // 3),
                 piece.shape_index, piece.rotation, piece.x)
        if len(state) != self.current_state_tuple_len: print("CRITICAL WARN: State len mismatch!"); state = state[:self.current_state_tuple_len] + (0,) * (self.current_state_tuple_len - len(state))
        return state

    def choose_action(self, state):
        if random.random() < self.epsilon: return random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            if np.all(q_values == 0): return random.choice(self.actions)
            else: return random.choice(np.where(q_values == np.max(q_values))[0])

    def learn(self, state, action, reward, next_state, game_over):
        current_q = self.q_table[state][action]
        target_q = reward if game_over else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] = current_q + self.lr * (target_q - current_q)

    def decay_exploration(self):
        if self.epsilon > self.min_epsilon: self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# --- Main Game Loop ---
def run_game(agent, train=False, num_episodes=1000, render=False, max_steps_per_episode=20000):
    pygame.init(); font = None
    try: font = pygame.font.SysFont(None, 25)
    except Exception as e: print(f"Warn: Font fail: {e}")
    screen = None; clock = pygame.time.Clock()
    if render: screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT)); pygame.display.set_caption('Tetris AI')

    session_high_score = 0; scores = []; total_steps_log = 0; episode_start_times = deque(maxlen=100)
    mode = "Train" if train else "Play"; render_stat = "Visible" if render else "Non-Visible"
    print(f"\n--- Start {mode} ({render_stat}) ---"); print(f"Start Eps: {agent.epsilon:.4f}")
    if train: print(f"Eps: {num_episodes}, Max Steps: {max_steps_per_episode}")
    else: print("Policy (min explore)."); agent.epsilon = agent.min_epsilon

    # Variables for Auto-Reset
    last_logged_avg_score = -np.inf
    stagnation_counter = 0
    STAGNATION_THRESHOLD = 3

    for episode in range(num_episodes):
        ep_start = time.time(); game = TetrisGame(); state_info = game.reset(); state = agent.get_state(state_info)
        game_over = False; steps = 0; ep_score = 0

        while not game_over and steps < max_steps_per_episode:
            pygame.event.pump()

            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Quit. Save...")
                        if train: agent.save_q_table()
                        pygame.quit(); return scores, session_high_score
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            agent.epsilon = agent.initial_epsilon
                            print(f"\n*** Epsilon reset to {agent.epsilon:.4f} by user! ***\n")
                            stagnation_counter = 0
                            last_logged_avg_score = -np.inf

            action = agent.choose_action(state)
            next_state_info, reward, game_over, current_score = game.step(action)
            ep_score = current_score

            if train:
                next_state = agent.get_state(next_state_info)
                if state_info["current_piece"] is not None:
                    agent.learn(state, action, reward, next_state, game_over)

            state_info = next_state_info; state = agent.get_state(state_info)

            steps += 1
            if train: total_steps_log += 1

            if render:
                game.draw(screen, font)
                if font:
                    ep_txt=font.render(f"E:{episode+1}/{num_episodes}",1,WHITE); eps_txt=font.render(f"Eps:{agent.epsilon:.4f}",1,WHITE); steps_txt=font.render(f"S:{steps}",1,WHITE)
                    screen.blit(ep_txt,(WINDOW_WIDTH-150,5)); screen.blit(eps_txt,(WINDOW_WIDTH-150,25)); screen.blit(steps_txt,(WINDOW_WIDTH-150,45))
                pygame.display.flip(); clock.tick(60)

        scores.append(ep_score); session_high_score = max(session_high_score, ep_score)
        ep_end = time.time(); episode_start_times.append(ep_end - ep_start); avg_ep_time = np.mean(episode_start_times) if episode_start_times else 0

        if train:
            agent.decay_exploration()
            log_interval = 100
            # Auto-Reset Logic inside Logging Block
            if (episode + 1) % log_interval == 0 or episode == num_episodes - 1:
                if len(scores) > 0:
                     current_avg_score = np.mean(scores[-log_interval:]) if len(scores)>=log_interval else np.mean(scores)
                else:
                     current_avg_score = 0

                print(f"E {episode+1}/{num_episodes}|AvgS({log_interval}):{current_avg_score:.0f}|HiS:{session_high_score}|Eps:{agent.epsilon:.4f}|Steps:{total_steps_log}|T/Ep:{avg_ep_time:.2f}s", end="")

                if last_logged_avg_score > -np.inf:
                    if current_avg_score <= last_logged_avg_score:
                        stagnation_counter += 1
                        print(f" (Stagnation: {stagnation_counter}/{STAGNATION_THRESHOLD})")
                    else:
                        stagnation_counter = 0
                        last_logged_avg_score = current_avg_score
                        print()
                else:
                    last_logged_avg_score = current_avg_score
                    print(" (Initial Log)")


                if stagnation_counter >= STAGNATION_THRESHOLD:
                    agent.epsilon = agent.initial_epsilon
                    print(f"*** Stagnation detected! Epsilon reset to {agent.epsilon:.4f}. ***")
                    stagnation_counter = 0
                    last_logged_avg_score = current_avg_score


                agent.save_q_table()
                total_steps_log = 0
            # End Auto-Reset Logic

        elif render: print(f"G {episode+1} Over! Scr:{ep_score}, Lines:{game.lines_cleared_total}, Steps:{steps}")


    if train: print("\n--- Train Finish ---"); agent.save_q_table()
    if render: pygame.quit()
    return scores, session_high_score

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or Run a Q-Learning Tetris AI.")
    parser.add_argument('mode', metavar='MODE', type=str, nargs='?', choices=['train', 'run'], default='run')
    parser.add_argument('-e', '--episodes', type=int, default=50000)
    parser.add_argument('-p', '--play-episodes', type=int, default=5)
    parser.add_argument('--render-train', action='store_true')
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--qtable', type=str, default="tetris.pkl")
    parser.add_argument('--max-steps', type=int, default=20000)
    parser.add_argument('--reset-epsilon', type=float, default=None) # Command-line reset option
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--decay', type=float, default=0.99995)
    args = parser.parse_args()

    DO_TRAINING=(args.mode=='train'); RENDER_GAME=(args.mode=='run') or args.render_train
    NUM_EPISODES=args.episodes if DO_TRAINING else args.play_episodes; LOAD_STATE= not args.fresh
    AGENT_FILE=args.qtable; MAX_STEPS=args.max_steps; file_exists=os.path.exists(AGENT_FILE)

    if args.mode=='run' and not file_exists: print(f"Error: Cannot run. '{AGENT_FILE}' not found."); exit(1)
    if DO_TRAINING and LOAD_STATE and not file_exists: print(f"Warn: Load requested but '{AGENT_FILE}' not found. Start fresh."); LOAD_STATE=False

    default_initial_epsilon = 1.0
    print_mode="Loading" if LOAD_STATE and file_exists else "Start fresh"
    print(f"{print_mode} agent state: '{AGENT_FILE}'")

    agent = QLearningAgent(learning_rate=args.lr, discount_factor=args.gamma,
                           exploration_rate=default_initial_epsilon,
                           exploration_decay=args.decay, min_exploration_rate=0.01,
                           q_table_file=AGENT_FILE)

    if LOAD_STATE and not agent.q_table and file_exists:
        print("Warn: Load failed or file empty. Resetting epsilon to initial.")
        agent.epsilon = agent.initial_epsilon

    if args.reset_epsilon is not None:
         print(f"!!! Force resetting epsilon via command line: {args.reset_epsilon:.4f} !!!")
         agent.epsilon = max(agent.min_epsilon, min(1.0, args.reset_epsilon))

    start_time = time.time(); high_score = 0; e_type = None; scores = []
    try: scores, high_score = run_game(agent, train=DO_TRAINING, num_episodes=NUM_EPISODES, render=RENDER_GAME, max_steps_per_episode=MAX_STEPS)
    except KeyboardInterrupt: print(f"\n{args.mode.capitalize()} interrupted."); e_type = KeyboardInterrupt
    except Exception: print(f"\nError:"); traceback.print_exc(); e_type = type(Exception)
    finally:
        if DO_TRAINING and e_type is not None:
            print("Save state on interrupt/error...")
            if agent: agent.save_q_table()

        end_time = time.time(); print(f"\n{args.mode.capitalize()} session finished."); print(f"Duration: {end_time - start_time:.2f}s.")
        if scores: print(f"Avg score ({len(scores)} games): {np.mean(scores):.2f}")
        print(f"Session High Score: {high_score}")
        if agent and (DO_TRAINING or (LOAD_STATE and file_exists)):
             final_eps=agent.epsilon; final_states=len(agent.q_table)
             print(f"Final state in '{AGENT_FILE}': {final_states} states, eps {final_eps:.4f}")
    print("\nScript finished.")