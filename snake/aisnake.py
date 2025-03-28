# aisnake.py
import pygame
import random
import numpy as np
from collections import deque, defaultdict
import pickle
import os
import time
import traceback
import argparse

WIDTH, HEIGHT = 600, 400
BLOCK_SIZE = 20
GRID_WIDTH = WIDTH // BLOCK_SIZE
GRID_HEIGHT = HEIGHT // BLOCK_SIZE
SPEED = 15

REWARD_EAT_FOOD = 60
REWARD_WALL_HIT = -50
REWARD_SELF_HIT = -120
REWARD_STEP_BASE = -0.05
REWARD_CLOSER_FOOD = 0.5
REWARD_FURTHER_FOOD = -0.5

NO_COLLISION = 0
WALL_COLLISION = 1
SELF_COLLISION = 2

WHITE = (255, 255, 255); BLACK = (0, 0, 0); RED = (213, 50, 80)
BLUE1 = (0, 0, 255); BLUE2 = (0, 100, 255)

UP = (0, -1); DOWN = (0, 1); LEFT = (-1, 0); RIGHT = (1, 0)

class Snake:
    def __init__(self):
        self.reset()

    def reset(self):
        self.length = 1
        start_x = random.randint(GRID_WIDTH // 4, 3 * GRID_WIDTH // 4)
        start_y = random.randint(GRID_HEIGHT // 4, 3 * GRID_HEIGHT // 4)
        self.positions = deque([(start_x, start_y)])
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.score = 0
        self.grow_pending = False

    def get_head_position(self):
        return self.positions[0]

    def turn(self, point):
        if self.length > 1 and (point[0] * -1, point[1] * -1) == self.direction:
            return
        else:
            self.direction = point

    def move(self):
        cur_head = self.get_head_position()
        dx, dy = self.direction
        new_head_x = cur_head[0] + dx
        new_head_y = cur_head[1] + dy
        new_head = (new_head_x, new_head_y)

        if not (0 <= new_head_x < GRID_WIDTH and 0 <= new_head_y < GRID_HEIGHT):
            return WALL_COLLISION

        body_check_positions = list(self.positions)
        check_against = body_check_positions if self.grow_pending else body_check_positions[:-1]
        if new_head in check_against:
             if not self.grow_pending and len(body_check_positions) > 1 and new_head == body_check_positions[-1]:
                 pass
             else:
                 return SELF_COLLISION

        self.positions.appendleft(new_head)

        if self.grow_pending:
            self.grow_pending = False
        else:
             if len(self.positions) > self.length:
                 self.positions.pop()

        return NO_COLLISION

    def grow(self):
        self.length += 1
        self.score += 1
        self.grow_pending = True

    def draw(self, surface):
        for i, p in enumerate(self.positions):
            r = pygame.Rect((p[0] * BLOCK_SIZE, p[1] * BLOCK_SIZE), (BLOCK_SIZE, BLOCK_SIZE))
            color = BLUE1 if i == 0 else BLUE2
            pygame.draw.rect(surface, color, r)
            pygame.draw.rect(surface, BLACK, r, 1)

class Food:
    def __init__(self):
        self.position = (0, 0)
        self.randomize_position(deque())

    def randomize_position(self, snake_positions):
        snake_pos_set = set(snake_positions)
        while True:
            self.position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if self.position not in snake_pos_set:
                break

    def draw(self, surface):
        r = pygame.Rect((self.position[0] * BLOCK_SIZE, self.position[1] * BLOCK_SIZE), (BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(surface, RED, r)
        pygame.draw.rect(surface, BLACK, r, 1)

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0,
                 exploration_decay=0.9995, min_exploration_rate=0.01, q_table_file='snake.pkl'):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.initial_epsilon = exploration_rate
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration_rate
        self.q_table_file = q_table_file
        self.actions = [UP, DOWN, LEFT, RIGHT]
        self.num_actions = len(self.actions)

        # --- Determine expected state length BEFORE loading ---
        dummy_snake = Snake()
        dummy_food = Food()
        # Ensure _get_state_internal is called to get the expected length based on the CURRENT code version
        self.current_state_len = len(self._get_state_internal(dummy_snake, dummy_food))
        print(f"State representation length: {self.current_state_len}") # Debug print

        self.q_table = self.load_q_table()


    def load_q_table(self):
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'rb') as f:
                    saved_data = pickle.load(f)

                if isinstance(saved_data, dict) and 'q_table' in saved_data and 'epsilon' in saved_data:
                    q_table_loaded = saved_data['q_table']
                    loaded_epsilon = saved_data['epsilon']
                    print(f"Loading agent state from '{self.q_table_file}'...")
                    self.epsilon = max(self.min_epsilon, min(1.0, loaded_epsilon)) # Clamp loaded epsilon
                    print(f" - Loaded Epsilon: {self.epsilon:.4f}")
                elif isinstance(saved_data, (dict, defaultdict)):
                    q_table_loaded = saved_data
                    print(f"Loading Q-table (old format) from '{self.q_table_file}'...")
                    print(f" - Resetting Epsilon to initial default: {self.initial_epsilon:.4f}")
                    self.epsilon = self.initial_epsilon
                else:
                    print(f"Error: Unknown data format in '{self.q_table_file}'. Starting fresh.")
                    self.epsilon = self.initial_epsilon
                    return defaultdict(lambda: np.zeros(self.num_actions))

                q_table = defaultdict(lambda: np.zeros(self.num_actions))
                valid_states_count = 0
                invalid_states_count = 0

                # --- Check state length compatibility ---
                first_state = next(iter(q_table_loaded), None) if q_table_loaded else None
                loaded_state_len = len(first_state) if isinstance(first_state, tuple) else -1

                if first_state is not None and loaded_state_len != self.current_state_len:
                    print(f"!! State length mismatch! Loaded state length: {loaded_state_len}, Expected: {self.current_state_len}. Discarding loaded Q-table. !!")
                    self.epsilon = self.initial_epsilon # Reset epsilon for fresh start
                    return defaultdict(lambda: np.zeros(self.num_actions)) # Return empty table

                # --- If compatible, load states ---
                print(f" - Loaded state length ({loaded_state_len}) matches expected ({self.current_state_len}). Proceeding with load.")
                for state, values in q_table_loaded.items():
                     q_val_array = np.zeros(self.num_actions)
                     valid_values = False
                     if isinstance(values, list) and len(values) == self.num_actions:
                         q_val_array = np.array(values); valid_values = True
                     elif isinstance(values, np.ndarray) and values.shape == (self.num_actions,):
                         q_val_array = values; valid_values = True

                     # Length check is redundant now, but keep format check
                     if valid_values and isinstance(state, tuple):
                        q_table[state] = q_val_array
                        valid_states_count += 1
                     elif valid_values:
                        invalid_states_count += 1 # Should not happen if length check passed

                print(f" - Loaded Q-Table States: {len(q_table_loaded)}")
                if invalid_states_count > 0: # Should be 0 if length check worked
                    print(f" - Warning: Discarded {invalid_states_count} states due to format mismatch.")
                print(f" - Kept {valid_states_count} compatible states.")

                if valid_states_count == 0 and len(q_table_loaded) > 0:
                     print(f" - Warning: Loaded file contained states, but none were compatible. Effectively starting fresh.")
                     self.epsilon = self.initial_epsilon

                return q_table

            except Exception as e:
                print(f"Error loading data from '{self.q_table_file}': {e}. Starting fresh.")
                self.epsilon = self.initial_epsilon
                return defaultdict(lambda: np.zeros(self.num_actions))
        else:
            print(f"Agent state file '{self.q_table_file}' not found. Starting fresh.")
            self.epsilon = self.initial_epsilon
            return defaultdict(lambda: np.zeros(self.num_actions))

    def save_q_table(self):
        try:
            data_to_save = {'q_table': dict(self.q_table), 'epsilon': self.epsilon}
            with open(self.q_table_file, 'wb') as f:
                pickle.dump(data_to_save, f)
        except Exception as e:
            print(f"Error saving data to '{self.q_table_file}': {e}")

    # --- UPDATED STATE REPRESENTATION ---
    def _get_state_internal(self, snake, food):
        head = snake.get_head_position()
        current_dir = snake.direction
        body_list = list(snake.positions)

        # Relative Directions
        if current_dir == UP:       dir_l, dir_s, dir_r = LEFT, UP, RIGHT
        elif current_dir == DOWN:   dir_l, dir_s, dir_r = RIGHT, DOWN, LEFT
        elif current_dir == LEFT:   dir_l, dir_s, dir_r = DOWN, LEFT, UP
        elif current_dir == RIGHT:  dir_l, dir_s, dir_r = UP, RIGHT, DOWN
        else: dir_l, dir_s, dir_r = LEFT, UP, RIGHT # Fallback

        # Danger Checks
        danger_straight = self._is_danger(head, dir_s, body_list, snake.length, steps=1)
        danger_left = self._is_danger(head, dir_l, body_list, snake.length, steps=1)
        danger_right = self._is_danger(head, dir_r, body_list, snake.length, steps=1)
        danger_straight_2 = self._is_danger(head, dir_s, body_list, snake.length, steps=2)

        # Food Direction
        food_x, food_y = food.position
        head_x, head_y = head
        food_dir_x = int(np.sign(food_x - head_x))
        food_dir_y = int(np.sign(food_y - head_y))

        # Relative Tail Direction
        tail_dir_x = 0
        tail_dir_y = 0
        if snake.length > 1:
            tail = snake.positions[-1]
            tail_x, tail_y = tail
            tail_dir_x = int(np.sign(tail_x - head_x))
            tail_dir_y = int(np.sign(tail_y - head_y))

        # Updated State Tuple
        state = (
            int(danger_straight),
            int(danger_left),
            int(danger_right),
            int(danger_straight_2),
            food_dir_x,
            food_dir_y,
            tail_dir_x, # New
            tail_dir_y, # New
            current_dir
        )
        return state
    # --- END OF UPDATED STATE ---

    def get_state(self, snake, food):
        # This now uses the updated _get_state_internal
        return self._get_state_internal(snake, food)

    def _is_danger(self, head, direction, body_list, snake_length, steps=1):
        point_x = head[0] + direction[0] * steps
        point_y = head[1] + direction[1] * steps
        check_pos = (point_x, point_y)

        if not (0 <= point_x < GRID_WIDTH and 0 <= point_y < GRID_HEIGHT):
            return True

        if steps == 1:
            if snake_length > 1 and check_pos in body_list[:-1]:
                return True
        elif steps > 1:
            if check_pos in body_list:
                return True

        return False

    def choose_action(self, state, snake):
        snake_direction = snake.direction

        if random.random() < self.epsilon:
            possible_action_indices = list(range(self.num_actions))

            if snake.length > 1:
                opposite_dir_tuple = (snake_direction[0] * -1, snake_direction[1] * -1)
                try:
                    opp_idx = self.actions.index(opposite_dir_tuple)
                    if opp_idx in possible_action_indices:
                        possible_action_indices.remove(opp_idx)
                except ValueError:
                    pass

            if not possible_action_indices:
                 try: action_index = self.actions.index(snake_direction)
                 except ValueError: action_index = random.choice(list(range(self.num_actions)))
            else:
                 action_index = random.choice(possible_action_indices)

        else:
            q_values = self.q_table[state]
            valid_action_indices = list(range(self.num_actions))

            if snake.length > 1:
                opposite_dir_tuple = (snake_direction[0] * -1, snake_direction[1] * -1)
                try:
                    opp_idx = self.actions.index(opposite_dir_tuple)
                    if opp_idx in valid_action_indices:
                        valid_action_indices.remove(opp_idx)
                except ValueError:
                    pass

            if not valid_action_indices:
                 try: best_action_index = self.actions.index(snake_direction)
                 except ValueError: best_action_index = random.choice(list(range(self.num_actions)))
            else:
                 best_q = -np.inf
                 best_indices = []
                 for i in valid_action_indices:
                     if q_values[i] > best_q:
                         best_q = q_values[i]
                         best_indices = [i]
                     elif q_values[i] == best_q:
                         best_indices.append(i)

                 if not best_indices:
                     best_action_index = random.choice(valid_action_indices)
                 else:
                     best_action_index = random.choice(best_indices)

            action_index = best_action_index

        return action_index

    def learn(self, state, action_index, reward, next_state):
        current_q = self.q_table[state][action_index]
        max_future_q = np.max(self.q_table[next_state])

        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)

        self.q_table[state][action_index] = new_q

    def decay_exploration(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)


# --- Game Loop Function (using the previous good auto-reset logic) ---
def run_game(agent, train=False, num_episodes=1000, render=False, max_steps_per_episode=5000,
             auto_reset_config=None):
    pygame.init()
    screen = None
    font = None
    clock = None

    if render:
        try:
            pygame.font.init()
            font = pygame.font.SysFont(None, 25)
        except Exception as e:
            print(f"Warning: Pygame font system failed to initialize: {e}. Score/info text disabled.")
            font = None
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Self-Learning Snake AI')
        clock = pygame.time.Clock()

    session_high_score = 0
    scores = []
    total_steps_session_log = 0

    mode = "Training" if train else "Playing"
    render_status = "Visible" if render else "Non-Visible"
    print(f"\n--- Starting {mode} ({render_status}) ---")
    print(f"Start Epsilon for this session: {agent.epsilon:.4f}")

    if train:
        print(f"Episodes: {num_episodes}, Max Steps/Ep: {max_steps_per_episode}")
    else:
        print("Using learned policy (minimal exploration).")
        agent.epsilon = agent.min_epsilon

    # --- Revised Auto-reset Stagnation Tracking ---
    best_avg_score_so_far = -np.inf
    stagnation_counter = 0
    log_interval = 100
    auto_reset_enabled = train and auto_reset_config is not None

    if auto_reset_enabled:
        print(f"Auto Epsilon Reset Enabled: Patience={auto_reset_config['patience']} intervals, ResetValue={auto_reset_config['value']:.2f}")

    for episode in range(num_episodes):
        snake = Snake()
        food = Food()
        food.randomize_position(snake.positions)

        game_over = False
        steps_taken = 0

        head_pos_init = snake.get_head_position()
        last_distance_to_food = np.linalg.norm(np.array(head_pos_init) - np.array(food.position))

        current_state = agent.get_state(snake, food)

        while not game_over and steps_taken < max_steps_per_episode:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nQuit signal received.")
                        if train:
                            print("Saving agent state before quitting...")
                            agent.save_q_table()
                        pygame.quit()
                        return scores, session_high_score
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r and train and auto_reset_enabled:
                            agent.epsilon = auto_reset_config['value']
                            print(f"\n*** Epsilon reset to {agent.epsilon:.4f} by user! ***\n")
                            stagnation_counter = 0
                            if len(scores) > 0:
                                current_avg_score_manual = np.mean(scores[-log_interval:]) if len(scores)>=log_interval else np.mean(scores)
                                best_avg_score_so_far = current_avg_score_manual
                            else:
                                best_avg_score_so_far = -np.inf


            action_index = agent.choose_action(current_state, snake)
            action_direction = agent.actions[action_index]

            snake.turn(action_direction)
            collision_type = snake.move()

            reward = REWARD_STEP_BASE
            ate_food = False

            if collision_type == WALL_COLLISION:
                game_over = True
                reward = REWARD_WALL_HIT
            elif collision_type == SELF_COLLISION:
                game_over = True
                reward = REWARD_SELF_HIT
            elif collision_type == NO_COLLISION:
                head_pos = snake.get_head_position()
                if head_pos == food.position:
                    snake.grow()
                    food.randomize_position(snake.positions)
                    reward = REWARD_EAT_FOOD
                    ate_food = True
                    last_distance_to_food = np.linalg.norm(np.array(head_pos) - np.array(food.position))
                else:
                    current_distance = np.linalg.norm(np.array(head_pos) - np.array(food.position))
                    if current_distance < last_distance_to_food:
                        if REWARD_CLOSER_FOOD != 0.0: reward += REWARD_CLOSER_FOOD
                    else:
                         if REWARD_FURTHER_FOOD != 0.0: reward += REWARD_FURTHER_FOOD
                    last_distance_to_food = current_distance

            next_state = agent.get_state(snake, food)

            if train:
                # Pass next_state directly to learn method
                agent.learn(current_state, action_index, reward, next_state)

            current_state = next_state
            steps_taken += 1
            if train: total_steps_session_log += 1

            if render:
                screen.fill(BLACK)
                snake.draw(screen)
                food.draw(screen)

                if font:
                    score_text = font.render(f"Score: {snake.score}", True, WHITE)
                    screen.blit(score_text, (5, 5))
                    high_score_text = font.render(f"High Score: {session_high_score}", True, WHITE)
                    screen.blit(high_score_text, (5, 30))
                    ep_text = font.render(f"Ep: {episode+1}/{num_episodes}", True, WHITE)
                    eps_text = font.render(f"Eps: {agent.epsilon:.3f}", True, WHITE)
                    screen.blit(ep_text, (WIDTH - 150, 5))
                    screen.blit(eps_text, (WIDTH - 150, 30))

                pygame.display.flip()
                clock.tick(SPEED)

        if snake.score > session_high_score:
            session_high_score = snake.score

        scores.append(snake.score)

        if train:
            agent.decay_exploration()

            if (episode + 1) % log_interval == 0:
                if len(scores) > 0:
                    current_avg_score = np.mean(scores[-log_interval:]) if len(scores)>=log_interval else np.mean(scores)
                else:
                    current_avg_score = 0.0

                print_msg = f"Ep {episode + 1}/{num_episodes} | AvgScore(last {log_interval}): {current_avg_score:.2f} | HighScore: {session_high_score} | Epsilon: {agent.epsilon:.4f} | Steps logged: {total_steps_session_log}"

                if auto_reset_enabled:
                    if best_avg_score_so_far == -np.inf and (episode + 1) >= log_interval:
                       best_avg_score_so_far = current_avg_score
                       print_msg += " | Initializing best avg score."

                    improvement = current_avg_score - best_avg_score_so_far
                    if improvement > 0.0:
                        print_msg += f" | New Best Avg! (+{improvement:.2f})"
                        best_avg_score_so_far = current_avg_score
                        stagnation_counter = 0
                    elif (episode + 1) > log_interval:
                        stagnation_counter += 1
                        print_msg += f" | Stagnated ({stagnation_counter}/{auto_reset_config['patience']})"

                        if stagnation_counter >= auto_reset_config['patience']:
                            old_epsilon = agent.epsilon
                            agent.epsilon = max(agent.min_epsilon, auto_reset_config['value'])
                            print_msg += f" | !!! AUTO-RESETTING Epsilon ({old_epsilon:.4f} -> {agent.epsilon:.4f}) !!!"
                            stagnation_counter = 0
                            best_avg_score_so_far = current_avg_score
                    else:
                         pass

                print(print_msg)
                agent.save_q_table()
                total_steps_session_log = 0

        if not train:
            print(f"Game {episode+1} finished! Score: {snake.score}, Steps: {steps_taken}")

    if train:
        print("\n--- Training Finished ---")
        agent.save_q_table()
        print(f"Final agent state saved to '{agent.q_table_file}'")

    if render:
        pygame.quit()

    return scores, session_high_score


# --- Main Execution Block (arguments are the same as previous version) ---
if __name__ == '__main__':
    DEFAULT_RESET_PATIENCE = 3
    DEFAULT_RESET_VALUE = 0.6

    parser = argparse.ArgumentParser(description="Train or Run a Q-Learning Snake AI with Auto Epsilon Reset.")

    parser.add_argument('mode', metavar='MODE', type=str, nargs='?', choices=['train', 'run'], default='run',
                        help="Operation mode: 'train' or 'run' (default: run).")
    parser.add_argument('-e', '--episodes', type=int, default=50000,
                        help="Number of episodes for training (default: 50000).")
    parser.add_argument('-p', '--play-episodes', type=int, default=5,
                        help="Number of episodes to watch during 'run' mode (default: 5).")
    parser.add_argument('--render-train', action='store_true',
                        help="Render the game window during training (significantly slower).")
    parser.add_argument('--fresh', action='store_true',
                        help="Force fresh training, ignoring any existing agent state file.")
    parser.add_argument('--qtable', type=str, default="snake.pkl",
                        help="Filename for saving/loading agent state (default: snake.pkl).")
    parser.add_argument('--max-steps', type=int, default=5000,
                        help="Maximum steps allowed per episode before termination (default: 5000).")
    parser.add_argument('--reset-epsilon', type=float, default=None,
                        help="Manually force epsilon to this value AFTER loading state. Applied at the start.")
    parser.add_argument('--auto-reset', action='store_true',
                        help="Enable automatic epsilon reset during training if performance stagnates.")
    parser.add_argument('--reset-patience', type=int, default=DEFAULT_RESET_PATIENCE,
                        help=f"Auto-Reset: Number of logging intervals (100 eps each) without a new best avg score before reset (default: {DEFAULT_RESET_PATIENCE}).")
    parser.add_argument('--reset-value', type=float, default=DEFAULT_RESET_VALUE,
                        help=f"Auto-Reset: Epsilon value to reset to (e.g., 0.6) (default: {DEFAULT_RESET_VALUE}).")

    args = parser.parse_args()

    DO_TRAINING = (args.mode == 'train')
    RENDER_GAME = (not DO_TRAINING) or args.render_train
    NUM_EPISODES_ARG = args.episodes if DO_TRAINING else args.play_episodes
    AGENT_STATE_FILENAME = args.qtable
    MAX_STEPS = args.max_steps
    LOAD_AGENT_STATE = not args.fresh

    auto_reset_params = None
    if DO_TRAINING and args.auto_reset:
        auto_reset_params = {
            'patience': args.reset_patience,
            'value': max(0.01, min(1.0, args.reset_value)),
        }
        if args.reset_epsilon is not None:
            print("Info: --reset-epsilon is set manually. Auto-reset logic will still activate later based on performance if enabled.")

    if args.mode == 'run' and not os.path.exists(AGENT_STATE_FILENAME):
        print(f"Error: Cannot run. Agent state file '{AGENT_STATE_FILENAME}' not found."); exit(1)

    # --- Modified File Handling for State Change ---
    if DO_TRAINING and LOAD_AGENT_STATE and os.path.exists(AGENT_STATE_FILENAME):
        print(f"Found existing agent file '{AGENT_STATE_FILENAME}'. Checking state format compatibility...")
        # The check is now done inside agent.load_q_table()
        pass # Proceed to agent initialization, which will handle loading/discarding
    elif DO_TRAINING and LOAD_AGENT_STATE and not os.path.exists(AGENT_STATE_FILENAME):
        print(f"Warning: Load requested (no --fresh flag) but agent file '{AGENT_STATE_FILENAME}' not found.")
        print("Starting fresh training.")
        LOAD_AGENT_STATE = False # Ensure fresh start if file not found
    elif DO_TRAINING and not LOAD_AGENT_STATE: # --fresh flag used
         print(f"Starting fresh training (ignoring '{AGENT_STATE_FILENAME}' due to --fresh).")


    print("-" * 30)
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=1.0,
        exploration_decay=0.9995,
        min_exploration_rate=0.01,
        q_table_file=AGENT_STATE_FILENAME
    )
    print("-" * 30)


    if args.reset_epsilon is not None:
         manual_eps_val = max(agent.min_epsilon, min(1.0, args.reset_epsilon))
         print(f"!!! MANUALLY Setting starting Epsilon to: {manual_eps_val:.4f} (overrides loaded/default) !!!")
         agent.epsilon = manual_eps_val


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
        if DO_TRAINING and e_type is not None:
             print("Attempting to save agent state due to interruption/error...")
             agent.save_q_table()
             if os.path.exists(agent.q_table_file):
                 print(f"Agent state saved to '{agent.q_table_file}'.")
             else:
                 print(f"Warning: Failed to save agent state to '{agent.q_table_file}' after interruption/error.")

        session_end_time = time.time()
        print(f"\n--- {args.mode.capitalize()} Session Summary ---")
        print(f"Session duration: {session_end_time - session_start_time:.2f} seconds.")

        if session_scores:
             print(f"Episodes completed: {len(session_scores)}")
             print(f"Average score: {np.mean(session_scores):.2f}")
             print(f"Highest score achieved: {session_high_score_final}")
        elif e_type is KeyboardInterrupt or e_type is not None :
             print("Session interrupted; partial or no episode scores recorded in final summary.")
        else:
             print("No episodes were completed.")

        if DO_TRAINING:
            print(f"Final Epsilon (at interrupt/end): {agent.epsilon:.4f}")
            # Use agent.q_table which exists even if loading failed and table is empty
            print(f"Total Q-Table states learned: {len(agent.q_table)}")


    print("\nScript finished.")