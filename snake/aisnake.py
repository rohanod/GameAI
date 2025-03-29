import pygame
import random
import numpy as np
from collections import deque, defaultdict
import pickle
import os
import time
import traceback
import argparse
import math # Added for potential future use (e.g., distance)

# --- Game Constants ---
WIDTH, HEIGHT = 480, 320
BLOCK_SIZE = 10
GRID_WIDTH = WIDTH // BLOCK_SIZE
GRID_HEIGHT = HEIGHT // BLOCK_SIZE
# SPEED = 15 # Less critical

# --- Multi-Instance Constants ---
NUM_INSTANCES = 9
GRID_ROWS = 3
GRID_COLS = 3

# --- Window Constants ---
INFO_HEIGHT = 60
INSTANCE_SPACING = 10
TOTAL_GRID_WIDTH = GRID_COLS * WIDTH + (GRID_COLS - 1) * INSTANCE_SPACING
TOTAL_GRID_HEIGHT = GRID_ROWS * HEIGHT + (GRID_ROWS - 1) * INSTANCE_SPACING
WINDOW_WIDTH = TOTAL_GRID_WIDTH
WINDOW_HEIGHT = TOTAL_GRID_HEIGHT + INFO_HEIGHT

# --- Rewards ---
# Increased food reward slightly, decreased step penalty slightly
REWARD_EAT_FOOD = 70
REWARD_WALL_HIT = -60 # Slightly increased penalty
REWARD_SELF_HIT = -120
REWARD_STEP_BASE = -0.02 # Reduced base penalty to encourage exploration slightly more
REWARD_CLOSER_FOOD = 0.6
REWARD_FURTHER_FOOD = -0.7 # Slightly stronger penalty for moving away

# --- Collision Types ---
NO_COLLISION = 0
WALL_COLLISION = 1
SELF_COLLISION = 2

# --- Colors ---
WHITE = (255, 255, 255); BLACK = (0, 0, 0); GRAY = (80, 80, 80)
RED = (213, 50, 80); BLUE1 = (0, 0, 255); BLUE2 = (0, 100, 255)

# --- Directions ---
UP = (0, -1); DOWN = (0, 1); LEFT = (-1, 0); RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT] # Keep a list accessible

# --- Action Mapping (Relative Turns) ---
# Actions are now relative: 0: Straight, 1: Turn Left, 2: Turn Right
# This simplifies the state space potentially, as the agent learns relative maneuvers.
ACTION_STRAIGHT = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
RELATIVE_ACTIONS = [ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT]
NUM_ACTIONS_RELATIVE = len(RELATIVE_ACTIONS)


class Snake:
    def __init__(self):
        self.reset()

    def reset(self):
        self.length = 1
        # Start closer to the center, less likely to hit wall immediately
        start_x = random.randint(GRID_WIDTH // 3, 2 * GRID_WIDTH // 3)
        start_y = random.randint(GRID_HEIGHT // 3, 2 * GRID_HEIGHT // 3)
        self.positions = deque([(start_x, start_y)])
        self.direction = random.choice(DIRECTIONS) # Initial absolute direction
        self.score = 0
        self.grow_pending = False

    def get_head_position(self):
        return self.positions[0]

    # *** Modified turn method to accept relative actions ***
    def turn(self, relative_action):
        """Turns the snake based on a relative action (straight, left, right)."""
        current_dx, current_dy = self.direction

        if relative_action == ACTION_STRAIGHT:
            # No change needed, direction remains the same
            pass
        elif relative_action == ACTION_LEFT:
            # Calculate new direction for a left turn
            new_direction = (current_dy, -current_dx) # 90-degree rotation matrix logic
            self.direction = new_direction
        elif relative_action == ACTION_RIGHT:
            # Calculate new direction for a right turn
            new_direction = (-current_dy, current_dx) # -90-degree rotation matrix logic
            self.direction = new_direction
        # Note: The check for reversing direction is implicitly handled
        # because the agent only chooses between straight, left, right, never 'back'.

    def move(self):
        cur_head = self.get_head_position()
        dx, dy = self.direction
        new_head_x = cur_head[0] + dx
        new_head_y = cur_head[1] + dy
        new_head = (new_head_x, new_head_y)

        # Wall collision check
        if not (0 <= new_head_x < GRID_WIDTH and 0 <= new_head_y < GRID_HEIGHT):
            return WALL_COLLISION

        # Self collision check (more careful check)
        # Check against all body parts *except* the very last one if not growing
        check_body = list(self.positions)
        # If we are not about to grow, the tail will move, so don't check collision with it.
        # If we *are* growing, the tail stays, so *do* check collision with it.
        comparison_body = check_body if self.grow_pending else check_body[:-1]
        if new_head in comparison_body:
             return SELF_COLLISION

        # Move snake
        self.positions.appendleft(new_head)

        if self.grow_pending:
            self.grow_pending = False # Growth happens now
        else:
             # Remove tail only if not growing
             if len(self.positions) > self.length:
                 self.positions.pop()

        return NO_COLLISION

    def grow(self):
        self.length += 1
        self.score += 1
        self.grow_pending = True # Mark that growth should occur *after* the next move

    def draw(self, surface, offset_x, offset_y):
        for i, p in enumerate(self.positions):
            draw_x = offset_x + p[0] * BLOCK_SIZE
            draw_y = offset_y + p[1] * BLOCK_SIZE
            r = pygame.Rect((draw_x, draw_y), (BLOCK_SIZE, BLOCK_SIZE))
            # Head is brighter blue, body darker
            color = (60, 120, 255) if i == 0 else (40, 80, 200) # Slightly different blues
            pygame.draw.rect(surface, color, r)
            pygame.draw.rect(surface, GRAY, r, 1) # Grid lines

class Food:
    def __init__(self):
        self.position = (0, 0)
        self.randomize_position(deque()) # Pass empty deque initially

    def randomize_position(self, snake_positions):
        snake_pos_set = set(snake_positions)
        available_pos = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if (x,y) not in snake_pos_set:
                    available_pos.append((x,y))

        if not available_pos:
            # Handle the (rare) case where the snake fills the board
            print("Warning: No space left for food!")
            # Default to a position, though the game is likely over
            self.position = (0,0)
        else:
             self.position = random.choice(available_pos)


    def draw(self, surface, offset_x, offset_y):
        draw_x = offset_x + self.position[0] * BLOCK_SIZE
        draw_y = offset_y + self.position[1] * BLOCK_SIZE
        r = pygame.Rect((draw_x, draw_y), (BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(surface, RED, r)
        pygame.draw.rect(surface, BLACK, r, 1)

class QLearningAgent:
    def __init__(self, learning_rate=0.1, lr_decay=0.99995, min_lr=0.01, # Added LR decay
                 discount_factor=0.95, # Slightly higher gamma
                 exploration_rate=1.0, exploration_decay=0.9995,
                 min_exploration_rate=0.01, q_table_file='snake.pkl'): # New filename
        self.lr = learning_rate
        self.initial_lr = learning_rate
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.gamma = discount_factor
        self.initial_epsilon = exploration_rate
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration_rate
        self.q_table_file = q_table_file
        self.actions = RELATIVE_ACTIONS # Use relative actions
        self.num_actions = NUM_ACTIONS_RELATIVE

        # Determine state length dynamically
        dummy_snake = Snake()
        dummy_food = Food()
        self.current_state_len = len(self._get_state_internal(dummy_snake, dummy_food))
        print(f"State representation length: {self.current_state_len}")

        self.q_table = self.load_q_table()

    def load_q_table(self):
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'rb') as f:
                    saved_data = pickle.load(f)

                # --- Enhanced Loading Logic ---
                q_table_loaded = {}
                loaded_epsilon = self.initial_epsilon
                loaded_lr = self.initial_lr
                valid_format = False

                if isinstance(saved_data, dict) and 'q_table' in saved_data:
                    q_table_loaded = saved_data.get('q_table', {})
                    loaded_epsilon = saved_data.get('epsilon', self.initial_epsilon)
                    loaded_lr = saved_data.get('lr', self.initial_lr) # Load LR if available
                    valid_format = True
                    print(f"Loading agent state from '{self.q_table_file}' (Format: New)")
                elif isinstance(saved_data, (dict, defaultdict)):
                    # Try loading old format (just the dict/defaultdict)
                    q_table_loaded = saved_data
                    valid_format = True
                    print(f"Loading Q-table from '{self.q_table_file}' (Format: Old). Epsilon/LR reset.")
                else:
                    print(f"Error: Unknown format in '{self.q_table_file}'. Starting fresh.")
                    return defaultdict(lambda: np.zeros(self.num_actions))

                # --- State Compatibility Check ---
                q_table = defaultdict(lambda: np.zeros(self.num_actions))
                valid_count = 0
                incompatible_count = 0
                malformed_count = 0

                if not q_table_loaded:
                     print(" - Q-Table in file is empty. Starting fresh.")
                     self.epsilon = self.initial_epsilon # Ensure reset if file was empty
                     self.lr = self.initial_lr
                     return q_table

                first_state_key = next(iter(q_table_loaded), None)

                if not isinstance(first_state_key, tuple):
                    print(f" !! Error: Q-table keys are not tuples. Discarding loaded data. !!")
                    return defaultdict(lambda: np.zeros(self.num_actions))

                loaded_state_len = len(first_state_key)

                if loaded_state_len != self.current_state_len:
                    print(f" !! State length mismatch! Loaded: {loaded_state_len}, Expected: {self.current_state_len}. Discarding loaded data. !!")
                    return defaultdict(lambda: np.zeros(self.num_actions))

                # --- Load Compatible States ---
                print(f" - State length OK ({loaded_state_len}). Loading states...")
                for state, values in q_table_loaded.items():
                    if isinstance(state, tuple) and len(state) == self.current_state_len:
                        q_vals = np.zeros(self.num_actions)
                        valid_values = False
                        if isinstance(values, list) and len(values) == self.num_actions:
                            q_vals = np.array(values)
                            valid_values = True
                        elif isinstance(values, np.ndarray) and values.shape == (self.num_actions,):
                            q_vals = values
                            valid_values = True

                        if valid_values:
                            q_table[state] = q_vals
                            valid_count += 1
                        else:
                            malformed_count += 1 # Values format issue
                    else:
                        incompatible_count +=1 # State format/length issue

                print(f" - Loaded Q-Table States (from file): {len(q_table_loaded)}")
                if incompatible_count > 0: print(f" - Warning: Discarded {incompatible_count} states due to format/length mismatch.")
                if malformed_count > 0: print(f" - Warning: Discarded {malformed_count} states due to value format mismatch.")
                print(f" - Successfully loaded {valid_count} compatible states.")

                if valid_count > 0:
                    self.epsilon = max(self.min_epsilon, min(1.0, loaded_epsilon))
                    self.lr = max(self.min_lr, min(self.initial_lr, loaded_lr)) # Use loaded LR
                    print(f" - Resuming with Epsilon: {self.epsilon:.4f}, LR: {self.lr:.4f}")
                else:
                    print(f" - Warning: No compatible states found. Starting fresh.")
                    self.epsilon = self.initial_epsilon
                    self.lr = self.initial_lr

                return q_table

            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Error loading '{self.q_table_file}' (corrupted?): {e}. Starting fresh.")
                return defaultdict(lambda: np.zeros(self.num_actions))
            except Exception as e:
                print(f"Unexpected error loading '{self.q_table_file}': {e}. Starting fresh.")
                traceback.print_exc()
                return defaultdict(lambda: np.zeros(self.num_actions))
        else:
            print(f"'{self.q_table_file}' not found. Starting fresh training.")
            return defaultdict(lambda: np.zeros(self.num_actions))

    def save_q_table(self):
        try:
            # Save Q-table, epsilon, and learning rate
            data_to_save = {
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'lr': self.lr
            }
            with open(self.q_table_file, 'wb') as f:
                pickle.dump(data_to_save, f)
            # print(f"Agent state saved to {self.q_table_file}") # Optional: uncomment for verbose saving
        except Exception as e:
            print(f"Error saving agent state: {e}")

    # *** NEW State Representation ***
    def _get_state_internal(self, snake, food):
        head = snake.get_head_position()
        hx, hy = head
        current_dir = snake.direction
        body_list = list(snake.positions) # Use list for easier slicing if needed
        snake_len = snake.length

        # 1. Define relative directions (straight, left, right)
        # Relative left/right calculated using 90-degree rotation logic
        dir_s = current_dir
        dir_l = (current_dir[1], -current_dir[0]) # Left relative to current_dir
        dir_r = (-current_dir[1], current_dir[0]) # Right relative to current_dir

        # 2. Check danger for potential next moves
        # Point straight ahead
        point_s = (hx + dir_s[0], hy + dir_s[1])
        danger_straight = self._is_danger(point_s, body_list, snake_len)

        # Point if turning left
        point_l = (hx + dir_l[0], hy + dir_l[1])
        danger_left = self._is_danger(point_l, body_list, snake_len)

        # Point if turning right
        point_r = (hx + dir_r[0], hy + dir_r[1])
        danger_right = self._is_danger(point_r, body_list, snake_len)

        # 3. Food direction relative to head
        food_x, food_y = food.position
        food_rel_x = food_x - hx
        food_rel_y = food_y - hy

        # Simple boolean flags for food location relative to head's orientation
        # This captures more immediate "is food left/right/ahead" info
        is_food_l = (dir_l[0] * food_rel_x + dir_l[1] * food_rel_y) > 0 # Dot product > 0 means food is generally in that direction
        is_food_r = (dir_r[0] * food_rel_x + dir_r[1] * food_rel_y) > 0
        is_food_s = (dir_s[0] * food_rel_x + dir_s[1] * food_rel_y) > 0

        # Optional: Add general direction as coarser info (can help if food is far)
        food_dir_x_sign = int(np.sign(food_rel_x))
        food_dir_y_sign = int(np.sign(food_rel_y))


        # State tuple: Danger (S/L/R), Food location relative (S/L/R bools), Food general direction (X/Y sign)
        # Total 8 features - keeps state space manageable but more informative than before.
        state = (
            int(danger_straight), int(danger_left), int(danger_right),
            int(is_food_s), int(is_food_l), int(is_food_r),
            food_dir_x_sign, food_dir_y_sign
        )
        # print(f"State: {state}") # DEBUG
        return state

    def get_state(self, snake, food):
         # Wrapper for potential future caching or modifications
        return self._get_state_internal(snake, food)

    def _is_danger(self, point, body_list, snake_length):
        """Checks if a given point is outside bounds or overlaps the snake body."""
        px, py = point
        # Check wall collision
        if not (0 <= px < GRID_WIDTH and 0 <= py < GRID_HEIGHT):
            return True
        # Check self collision
        # Need to check against all body parts. The 'move' logic handles the tail exception.
        # Here, we predict if the *next* chosen spot is already occupied (or will be by a non-moving tail if growing)
        if point in body_list:
             # If snake length is 1, it can't collide with itself yet
             if snake_length > 1:
                 return True # Simplified: any overlap is danger

        return False

    # Choose relative action (0=straight, 1=left, 2=right)
    def choose_action(self, state, snake): # Pass snake for context if needed (length > 1 check isn't relevant here anymore)
        if random.random() < self.epsilon:
            # Explore: choose a random relative action
            action_index = random.choice(self.actions) # actions are [0, 1, 2]
        else:
            # Exploit: choose the best known relative action
            q_values = self.q_table[state]
            # Find the action index(es) with the maximum Q-value
            max_q = np.max(q_values)
            # Get indices of all actions that have the max Q-value
            best_indices = [i for i, q in enumerate(q_values) if q == max_q]
            # Choose randomly among the best actions (breaks ties)
            action_index = random.choice(best_indices)

        return action_index # Returns 0, 1, or 2

    def learn(self, state, action_index, reward, next_state):
        """Update Q-table using the Q-learning formula."""
        current_q = self.q_table[state][action_index]
        # Find the best Q-value for the *next* state
        max_future_q = np.max(self.q_table[next_state]) # Standard Q-learning

        # Q-learning formula
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action_index] = new_q

        # --- Learning Rate Decay ---
        if self.lr > self.min_lr:
            self.lr *= self.lr_decay
            self.lr = max(self.min_lr, self.lr) # Clamp to minimum

    def decay_exploration(self):
        """Decay the exploration rate (epsilon) over time."""
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon) # Clamp to minimum


# Helper function for grid layout (unchanged)
def get_instance_offset(instance_index):
    row = instance_index // GRID_COLS
    col = instance_index % GRID_COLS
    offset_x = col * (WIDTH + INSTANCE_SPACING)
    offset_y = INFO_HEIGHT + row * (HEIGHT + INSTANCE_SPACING)
    return offset_x, offset_y

# --- Game Loop Function (Adapted for Relative Actions and New State) ---
def run_game(agent, train=False, num_episodes=1000, render=False, max_steps_per_episode=5000,
             auto_reset_config=None):
    pygame.init()
    screen = None; font = None; clock = None
    if render:
        try: font = pygame.font.SysFont(None, 24) # Slightly larger font
        except Exception as e: print(f"Warning: Font init failed: {e}.")
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(f'Multi-Snake AI ({NUM_INSTANCES} instances) - Smarter')
        clock = pygame.time.Clock()

    # Initialize instances
    snakes = [Snake() for _ in range(NUM_INSTANCES)]
    foods = [Food() for _ in range(NUM_INSTANCES)]
    for i in range(NUM_INSTANCES): foods[i].randomize_position(snakes[i].positions) # Ensure food isn't on snake start
    current_states = [agent.get_state(snakes[i], foods[i]) for i in range(NUM_INSTANCES)]
    last_distances = [np.linalg.norm(np.array(snakes[i].get_head_position()) - np.array(foods[i].position)) for i in range(NUM_INSTANCES)]
    instance_scores = [0] * NUM_INSTANCES
    instance_steps = [0] * NUM_INSTANCES
    instance_game_over = [False] * NUM_INSTANCES

    # Global tracking
    session_high_score = 0
    completed_episode_scores = []
    total_steps_log = 0
    global_step_counter = 0

    mode = "Training" if train else "Playing"; render_stat = "Visible" if render else "Non-Visible"
    print(f"\n--- Starting {mode} ({render_stat}) for {NUM_INSTANCES} instances ---")
    print(f"Start Epsilon: {agent.epsilon:.4f}, Start LR: {agent.lr:.4f}")
    if train: print(f"Episodes target: {num_episodes}, Max Steps/Reset: {max_steps_per_episode}")
    else: print("Using learned policy (min explore)."); agent.epsilon = agent.min_epsilon

    # Auto-reset tracking
    best_avg_score_so_far = -np.inf # Initialize properly
    high_score_at_last_log = 0
    stagnation_counter = 0
    # Log more frequently maybe? Adjust based on training speed.
    # Let's keep it relative to num_instances.
    log_interval_episodes = max(NUM_INSTANCES, 100) * (NUM_INSTANCES // 3 + 1) # Scale log interval
    last_log_episode_count = 0
    auto_reset_enabled = train and auto_reset_config is not None
    if auto_reset_enabled:
         print(f"Auto Reset: Patience={auto_reset_config['patience']} intervals, ResetValue={auto_reset_config['value']:.2f}")
         # Initialize baseline score from loaded data if possible
         # Note: This is tricky as 'average score' depends on past runs not saved.
         # We'll initialize based on the first interval's performance.


    # --- Main Loop ---
    completed_episodes_total = 0
    quit_signal = False
    last_save_time = time.time()
    save_interval_seconds = 300 # Save every 5 minutes during training

    while completed_episodes_total < num_episodes and not quit_signal:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: quit_signal = True; break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and train and auto_reset_enabled:
                        old_eps = agent.epsilon
                        agent.epsilon = max(agent.min_epsilon, auto_reset_config['value'])
                        print(f"\n*** Epsilon MANUALLY reset to {agent.epsilon:.4f} (was {old_eps:.4f})! Stagnation reset. ***\n")
                        stagnation_counter = 0 # Reset counter
                        # Reset the baseline average score based on recent performance to avoid immediate re-trigger
                        if completed_episode_scores:
                             relevant_scores = completed_episode_scores[-log_interval_episodes:]
                             best_avg_score_so_far = np.mean(relevant_scores) if relevant_scores else -np.inf
                        else: best_avg_score_so_far = -np.inf
                        high_score_at_last_log = session_high_score # Update baseline high score too
            if quit_signal: break

        active_instances_this_step = 0
        for i in range(NUM_INSTANCES):
            if instance_game_over[i]: continue

            active_instances_this_step += 1
            current_state = current_states[i]
            snake = snakes[i]
            food = foods[i]

            # Agent chooses a *relative* action (0, 1, or 2)
            relative_action_index = agent.choose_action(current_state, snake)

            # Instance takes step based on the relative action
            snake.turn(relative_action_index) # Turn first based on relative action
            collision_type = snake.move()       # Then move in the new direction

            # Calculate reward
            reward = REWARD_STEP_BASE
            game_over_this_step = False
            head_pos = snake.get_head_position() # Get head pos *after* move

            if collision_type == WALL_COLLISION:
                game_over_this_step = True
                reward += REWARD_WALL_HIT # Use += to include base penalty
            elif collision_type == SELF_COLLISION:
                game_over_this_step = True
                reward += REWARD_SELF_HIT # Use +=
            elif collision_type == NO_COLLISION:
                if head_pos == food.position:
                    snake.grow()
                    food.randomize_position(snake.positions)
                    reward += REWARD_EAT_FOOD
                    # Reset distance calculation after eating
                    last_distances[i] = np.linalg.norm(np.array(head_pos) - np.array(food.position))
                else:
                    # Reward moving closer/further from food
                    current_distance = np.linalg.norm(np.array(head_pos) - np.array(food.position))
                    if current_distance < last_distances[i]:
                        reward += REWARD_CLOSER_FOOD
                    elif current_distance > last_distances[i]: # Only penalize if actually moved further away
                        reward += REWARD_FURTHER_FOOD
                    last_distances[i] = current_distance

            # Check for max steps timeout
            instance_steps[i] += 1
            if not game_over_this_step and instance_steps[i] >= max_steps_per_episode:
                game_over_this_step = True
                reward -= 20 # Small penalty for timeout (optional)

            # Get next state *after* the move and potential food respawn
            next_state = agent.get_state(snake, food)

            # Agent learns (if training)
            if train:
                # Pass the chosen relative_action_index
                agent.learn(current_state, relative_action_index, reward, next_state)

            # Update instance state for the next iteration
            current_states[i] = next_state
            instance_scores[i] = snake.score # Track score for display
            total_steps_log += 1
            global_step_counter += 1

            # Handle Game Over for this instance
            if game_over_this_step:
                instance_game_over[i] = True
                completed_episode_scores.append(snake.score)
                session_high_score = max(session_high_score, snake.score)
                completed_episodes_total += 1

                # --- Logging and Auto-Reset Check ---
                # Check if enough episodes completed *since the last log* OR if target met
                if train and (completed_episodes_total >= num_episodes or (completed_episodes_total - last_log_episode_count) >= log_interval_episodes):
                    episodes_in_interval = completed_episodes_total - last_log_episode_count
                    if episodes_in_interval > 0: # Ensure we don't log empty intervals
                        relevant_scores = completed_episode_scores[-episodes_in_interval:]
                        current_avg_score = np.mean(relevant_scores) if relevant_scores else 0.0

                        print_msg = (f"Total Eps {completed_episodes_total}/{num_episodes} | "
                                     f"AvgScore(last {episodes_in_interval}): {current_avg_score:.2f} | "
                                     f"HighScore: {session_high_score} | "
                                     f"Eps: {agent.epsilon:.4f} | LR: {agent.lr:.5f} | "
                                     f"Steps logged: {total_steps_log} | QSize: {len(agent.q_table)}")

                        if auto_reset_enabled:
                            # Check for improvement based on *average* score and *high* score
                            new_high_score_achieved = (session_high_score > high_score_at_last_log)
                            significant_avg_improvement = (current_avg_score > best_avg_score_so_far + 0.1) # Require a small threshold increase

                            if new_high_score_achieved:
                                print_msg += " | New High Score!"
                                stagnation_counter = 0 # Reset stagnation
                                best_avg_score_so_far = max(best_avg_score_so_far, current_avg_score) # Update best avg too
                            elif significant_avg_improvement:
                                print_msg += f" | Improved Avg! (+{current_avg_score - best_avg_score_so_far:.2f})"
                                best_avg_score_so_far = current_avg_score # Update baseline
                                stagnation_counter = 0 # Reset stagnation
                            # Only start checking stagnation after the first interval completes fully
                            # or if best_avg_score_so_far was initialized to something other than -inf (e.g. from load)
                            elif last_log_episode_count > 0 or best_avg_score_so_far > -np.inf :
                                stagnation_counter += 1
                                print_msg += f" | Stagnated ({stagnation_counter}/{auto_reset_config['patience']})"
                                if stagnation_counter >= auto_reset_config['patience']:
                                    old_eps = agent.epsilon
                                    agent.epsilon = max(agent.min_epsilon, auto_reset_config['value'])
                                    print_msg += f" | !!! AUTO-RESET Epsilon ({old_eps:.4f} -> {agent.epsilon:.4f}) !!!"
                                    stagnation_counter = 0 # Reset counter
                                    # Reset baseline average to current average to avoid immediate re-trigger
                                    best_avg_score_so_far = current_avg_score
                            else: # First interval completed
                                print_msg += " | Baseline Avg Set."
                                best_avg_score_so_far = current_avg_score # Initialize baseline

                            high_score_at_last_log = session_high_score # Update high score baseline for next check

                        print(print_msg)
                        # Save periodically and at log points
                        agent.save_q_table()
                        last_save_time = time.time()
                        total_steps_log = 0 # Reset step counter for the next interval
                        last_log_episode_count = completed_episodes_total # Update log marker

            # Early exit if total episodes reached after processing this instance
            if completed_episodes_total >= num_episodes:
                 break # Break inner loop

        # --- End Instance Loop ---

        # --- Reset Finished Instances ---
        for i in range(NUM_INSTANCES):
            if instance_game_over[i]:
                snakes[i].reset()
                foods[i].randomize_position(snakes[i].positions) # Ensure valid food placement
                current_states[i] = agent.get_state(snakes[i], foods[i])
                last_distances[i] = np.linalg.norm(np.array(snakes[i].get_head_position()) - np.array(foods[i].position))
                instance_scores[i] = 0
                instance_steps[i] = 0
                instance_game_over[i] = False # Ready for next round

        # --- Rendering ---
        if render:
            screen.fill(BLACK)
            # Instance rendering
            for i in range(NUM_INSTANCES):
                offset_x, offset_y = get_instance_offset(i)
                # Background for each instance grid
                pygame.draw.rect(screen, (20, 20, 20), (offset_x, offset_y, WIDTH, HEIGHT))
                # Border
                pygame.draw.rect(screen, GRAY, (offset_x - 1, offset_y - 1, WIDTH + 2, HEIGHT + 2), 1)

                snakes[i].draw(screen, offset_x, offset_y)
                foods[i].draw(screen, offset_x, offset_y)
                # Instance score display
                if font:
                     inst_score_txt = font.render(f"{i}: {instance_scores[i]} ({instance_steps[i]})", True, WHITE)
                     screen.blit(inst_score_txt, (offset_x + 5, offset_y + 5))

            # Global info rendering (Top Bar)
            if font:
                # Calculate live average score across active instances
                active_scores = [s.score for idx, s in enumerate(snakes)]
                live_avg_score = np.mean(active_scores) if active_scores else 0.0

                # Create text surfaces
                texts = [
                    f"Total Eps: {completed_episodes_total}/{num_episodes}",
                    f"Total Steps: {global_step_counter}",
                    f"Live Avg Score: {live_avg_score:.2f}",
                    f"Session High: {session_high_score}",
                    f"Epsilon: {agent.epsilon:.4f}",
                    f"LR: {agent.lr:.5f}",
                    f"Q-States: {len(agent.q_table)}"
                ]
                # Position texts
                y_offset = 5
                x_offset1 = 5
                x_offset2 = WINDOW_WIDTH - 280 # Adjust as needed

                for idx, txt in enumerate(texts):
                    surf = font.render(txt, True, WHITE)
                    x_pos = x_offset2 if idx >= 3 else x_offset1 # Split info left/right
                    screen.blit(surf, (x_pos, y_offset + (idx % 3) * 20))

            pygame.display.flip()
            if clock: clock.tick(60) # Maintain consistent render speed

        # --- Decay Epsilon (once per global step) ---
        if train:
            agent.decay_exploration()

        # --- Periodic Saving (Independent of Logging) ---
        current_time = time.time()
        if train and (current_time - last_save_time > save_interval_seconds):
            print(f"\n--- Periodic Save ({(current_time - last_save_time):.0f}s elapsed) ---")
            agent.save_q_table()
            last_save_time = current_time
            print(f"Agent state saved. Next save in ~{save_interval_seconds}s.")

    # --- End Main Loop ---

    if quit_signal and train:
        print("Quit signal received. Saving final agent state...")
        agent.save_q_table()
    elif train:
        print("\n--- Training Finished ---")
        agent.save_q_table()
        print(f"Final agent state saved to {agent.q_table_file}")

    if render: pygame.quit()
    return completed_episode_scores, session_high_score


# --- Main Execution Block ---
if __name__ == '__main__':
    DEFAULT_RESET_PATIENCE = 6 # Slightly more patient by default
    DEFAULT_RESET_VALUE = 0.5 # Default reset epsilon value

    parser = argparse.ArgumentParser(description="Train/Run Multi-Snake AI (Smarter Version).")
    parser.add_argument('mode', metavar='MODE', type=str, nargs='?', choices=['train', 'run'], default='run', help="Mode: 'train' or 'run'")
    parser.add_argument('-e', '--episodes', type=int, default=100000, help="Total training episodes across all instances.") # Increased default
    parser.add_argument('-p', '--play-episodes', type=int, default=NUM_INSTANCES, help="Number of games to play in 'run' mode.") # Default to one full grid
    parser.add_argument('--render-train', action='store_true', help="Render the game grid during training (can slow down training significantly).")
    parser.add_argument('--fresh', action='store_true', help="Start training from scratch, ignore existing Q-table.")
    parser.add_argument('--qtable', type=str, default="snake.pkl", help="Filename for saving/loading the Q-table and agent state.")
    parser.add_argument('--max-steps', type=int, default=GRID_WIDTH * GRID_HEIGHT * 2, help="Max steps per episode before auto-resetting instance.") # Scaled default
    parser.add_argument('--reset-epsilon', type=float, default=None, help="Manually set the starting epsilon value (overrides loaded value).")
    # Auto-reset arguments
    parser.add_argument('--auto-reset', action='store_true', help="Enable automatic epsilon reset if training stagnates.")
    parser.add_argument('--reset-patience', type=int, default=DEFAULT_RESET_PATIENCE, help="Number of log intervals without improvement before resetting epsilon.")
    parser.add_argument('--reset-value', type=float, default=DEFAULT_RESET_VALUE, help="Epsilon value to reset to upon stagnation.")

    args = parser.parse_args()

    DO_TRAINING = (args.mode == 'train')
    RENDER_GAME = (args.mode == 'run') or args.render_train
    NUM_EPISODES_ARG = args.episodes if DO_TRAINING else args.play_episodes
    AGENT_STATE_FILENAME = args.qtable
    MAX_STEPS = args.max_steps
    LOAD_AGENT_STATE = not args.fresh # Load unless --fresh is specified

    auto_reset_params = None
    if DO_TRAINING and args.auto_reset:
        auto_reset_params = {
            'patience': max(1, args.reset_patience),
            'value': max(0.01, min(1.0, args.reset_value)) # Clamp reset value
         }
        print("Auto-reset enabled.")
        if args.reset_epsilon is not None:
            print("Note: --reset-epsilon is set, but auto-reset logic will take over if triggered during training.")

    # Input validation and info messages
    if args.mode == 'run' and not os.path.exists(AGENT_STATE_FILENAME):
        print(f"Error: Run mode selected, but required agent file '{AGENT_STATE_FILENAME}' not found.")
        exit(1)
    if DO_TRAINING and LOAD_AGENT_STATE and not os.path.exists(AGENT_STATE_FILENAME):
        print(f"Warning: Load requested but '{AGENT_STATE_FILENAME}' not found. Starting fresh training.")
        LOAD_AGENT_STATE = False # Force fresh if file not found
    elif DO_TRAINING and not LOAD_AGENT_STATE:
        print(f"Starting fresh training (--fresh specified or no existing agent file).")
    elif DO_TRAINING and LOAD_AGENT_STATE:
         print(f"Attempting to load agent state from '{AGENT_STATE_FILENAME}'.")

    print("-" * 30)
    # Initialize Agent (pass new LR params)
    agent = QLearningAgent(
        learning_rate=0.1,       # Initial LR
        lr_decay=0.99998,        # Slower decay maybe better?
        min_lr=0.005,            # Lower min LR
        discount_factor=0.95,    # Gamma
        exploration_rate=1.0,    # Initial Epsilon
        exploration_decay=0.9998, # Slower epsilon decay allows longer exploration
        min_exploration_rate=0.01, # Min Epsilon
        q_table_file=AGENT_STATE_FILENAME
    )
    print("-" * 30)

    # Manual Epsilon Override (after loading attempt)
    if args.reset_epsilon is not None:
         manual_eps_val = max(agent.min_epsilon, min(1.0, args.reset_epsilon))
         print(f"!!! MANUALLY Setting starting Epsilon to: {manual_eps_val:.4f} (overrides loaded value) !!!")
         agent.epsilon = manual_eps_val

    # --- Run the Game ---
    session_start_time = time.time()
    session_high_score_final = 0
    session_scores = []
    e_type = None

    try:
        session_scores, session_high_score_final = run_game(
            agent,
            train=DO_TRAINING,
            num_episodes=NUM_EPISODES_ARG,
            render=RENDER_GAME,
            max_steps_per_episode=MAX_STEPS,
            auto_reset_config=auto_reset_params
        )
    except KeyboardInterrupt:
        print(f"\n{args.mode.capitalize()} interrupted by user (Ctrl+C).")
        e_type = KeyboardInterrupt
    except Exception as e:
        print(f"\nAn unexpected error occurred during {args.mode}: {e}")
        traceback.print_exc()
        e_type = type(e)
    finally:
        # Attempt to save state if training was interrupted or errored
        if DO_TRAINING and e_type is not None:
             print("Attempting to save agent state on exit...")
             agent.save_q_table()
             if os.path.exists(agent.q_table_file):
                 print(f"Agent state saved successfully to '{agent.q_table_file}'.")
             else:
                 print(f"Warning: Failed to save agent state to '{agent.q_table_file}'.")

        # Print Session Summary
        session_end_time = time.time()
        duration = session_end_time - session_start_time
        print(f"\n--- {args.mode.capitalize()} Session Summary ---")
        print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

        if session_scores:
            total_completed_eps = len(session_scores)
            avg_score = np.mean(session_scores)
            median_score = np.median(session_scores)
            print(f"Total Episodes Completed: {total_completed_eps}")
            print(f"Average Score: {avg_score:.2f}")
            print(f"Median Score: {median_score:.2f}")
            print(f"Session High Score: {session_high_score_final}")
        elif e_type is KeyboardInterrupt:
            print("Session interrupted early; final scores may be incomplete.")
        elif e_type is not None:
             print("Session terminated due to error; no final scores recorded.")
        else:
            print("No episodes completed.")

        if DO_TRAINING:
            print(f"Final Epsilon: {agent.epsilon:.4f}")
            print(f"Final Learning Rate: {agent.lr:.5f}")
            print(f"Q-Table Size (Number of States Explored): {len(agent.q_table)}")

    print("\nScript finished.")