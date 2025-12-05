"""
Project Aether: Autonomous Atmospheric Entry System
---------------------------------------------------
A pure-Python implementation of Neuro-Evolutionary Reinforcement Learning.

Features:
- Custom Physics Engine (Mars-like gravity, atmospheric turbulence).
- From-Scratch Neural Network (Dense architecture, no ML libs).
- Adaptive Genetic Algorithm (Dynamic mutation rates, elitism).
- Real-time Matplotlib Telemetry Dashboard.

Dependencies: numpy, matplotlib
Usage: python aether_lander_ai.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import copy
import random
import time
from typing import List, Tuple, Optional

# --- CONFIGURATION & HYPERPARAMETERS ---
CONFIG = {
    'GRAVITY': -3.71,        # m/s^2 (Mars-like)
    'DT': 0.05,              # Time step (seconds)
    'MAX_FUEL': 200.0,       # Units of fuel
    'MAX_THRUST': 12.0,      # m/s^2 acceleration
    'WIND_STRENGTH': 2.5,    # Max random wind force
    'LANDING_TARGET': (0, 0),# x, y coordinates
    'LANDING_TOLERANCE': 2.0,# Distance to be considered "landed"
    'MAX_VELOCITY': 4.0,     # Max safe landing velocity
    'POPULATION_SIZE': 50,   # Agents per generation
    'GENERATIONS': 200,      # Max generations
    'MUTATION_RATE': 0.1,    # Base mutation chance
    'HIDDEN_LAYERS': [16, 12] # Neural Net Architecture
}

# --- 1. THE PHYSICS ENGINE ---

class LanderState:
    def __init__(self, x, y, vx, vy, fuel):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.fuel = fuel
        self.angle = 0.0  # Radians
        self.landed = False
        self.crashed = False
        self.trajectory = [] # History for plotting

    def update(self, thrust_power: float, rotate_cmd: float, dt: float):
        if self.landed or self.crashed:
            return

        self.trajectory.append((self.x, self.y))

        # Fuel consumption
        thrust_power = np.clip(thrust_power, 0, 1)
        if self.fuel <= 0:
            thrust_power = 0
        
        self.fuel -= thrust_power * 0.5 * dt

        # Orientation physics (simplified)
        self.angle += rotate_cmd * 2.0 * dt
        
        # Forces
        # Thrust components
        acc_x = -np.sin(self.angle) * thrust_power * CONFIG['MAX_THRUST']
        acc_y = np.cos(self.angle) * thrust_power * CONFIG['MAX_THRUST']
        
        # Gravity
        acc_y += CONFIG['GRAVITY']
        
        # Wind / Turbulence (Stochastic element)
        if self.y > 10: # Wind is stronger higher up
            wind_noise = np.random.uniform(-1, 1) * CONFIG['WIND_STRENGTH']
            acc_x += wind_noise

        # Integration (Euler)
        self.vx += acc_x * dt
        self.vy += acc_y * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Ground Collision Check
        if self.y <= 0:
            self.y = 0
            dist_to_target = np.abs(self.x - CONFIG['LANDING_TARGET'][0])
            velocity_mag = np.sqrt(self.vx**2 + self.vy**2)
            
            if (dist_to_target < CONFIG['LANDING_TOLERANCE'] and 
                velocity_mag < CONFIG['MAX_VELOCITY'] and 
                np.abs(self.angle) < 0.5):
                self.landed = True
            else:
                self.crashed = True

# --- 2. THE BRAIN (NEURAL NETWORK) ---

class NeuralNetwork:
    """
    A dense, feed-forward neural network implemented with raw NumPy.
    Input -> Hidden Layers -> Output
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layers = []
        # Input to First Hidden
        self.layers.append(np.random.randn(input_size, hidden_sizes[0]))
        self.biases = [np.random.randn(hidden_sizes[0])]
        
        # Hidden to Hidden
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(np.random.randn(hidden_sizes[i], hidden_sizes[i+1]))
            self.biases.append(np.random.randn(hidden_sizes[i+1]))
            
        # Hidden to Output
        self.layers.append(np.random.randn(hidden_sizes[-1], output_size))
        self.biases.append(np.random.randn(output_size))

    def forward(self, x):
        """Forward pass with Tanh activation (allows negative output for rotation)"""
        out = x
        for i, (weights, bias) in enumerate(zip(self.layers, self.biases)):
            out = np.dot(out, weights) + bias
            # Last layer uses Tanh for range [-1, 1], others ReLU or Tanh
            if i == len(self.layers) - 1:
                out = np.tanh(out) 
            else:
                out = np.tanh(out) # Tanh works well for control tasks
        return out

    def mutate(self, rate=0.1, intensity=0.5):
        """Applies Gaussian noise to weights and biases to simulate genetic mutation."""
        for i in range(len(self.layers)):
            # Create a mask for sparse mutation
            mask_w = np.random.rand(*self.layers[i].shape) < rate
            noise_w = np.random.randn(*self.layers[i].shape) * intensity
            self.layers[i] += mask_w * noise_w
            
            mask_b = np.random.rand(*self.biases[i].shape) < rate
            noise_b = np.random.randn(*self.biases[i].shape) * intensity
            self.biases[i] += mask_b * noise_b

# --- 3. THE ORGANISM (AGENT) ---

class Agent:
    def __init__(self):
        # 6 Inputs: x, y, vx, vy, angle, fuel
        # 2 Outputs: Main Thrust (0-1), Rotate (-1 to 1)
        self.brain = NeuralNetwork(6, CONFIG['HIDDEN_LAYERS'], 2)
        self.fitness = 0.0

    def evaluate(self, initial_x, initial_y):
        state = LanderState(initial_x, initial_y, 0, 0, CONFIG['MAX_FUEL'])
        
        steps = 0
        max_steps = 600 # 30 seconds max flight time
        
        while not (state.landed or state.crashed) and steps < max_steps:
            # Normalize inputs for better NN performance
            inputs = np.array([
                state.x / 50.0,
                state.y / 50.0,
                state.vx / 10.0,
                state.vy / 10.0,
                state.angle,
                state.fuel / CONFIG['MAX_FUEL']
            ])
            
            action = self.brain.forward(inputs)
            
            # Action [0] is thrust (-1 to 1) -> map to 0 to 1
            thrust = (action[0] + 1) / 2
            rotate = action[1] # -1 to 1
            
            state.update(thrust, rotate, CONFIG['DT'])
            steps += 1
            
        self.calculate_fitness(state, steps)
        return state # Return final state for vis

    def calculate_fitness(self, state, steps):
        """
        Fitness Function: The crucial driver of evolution.
        Reward landing, penalize crashing, reward fuel efficiency and closeness.
        """
        dist_x = np.abs(state.x - CONFIG['LANDING_TARGET'][0])
        dist_y = np.abs(state.y - CONFIG['LANDING_TARGET'][1])
        
        # Base score based on distance (closer is better)
        score = 100 - (dist_x + dist_y)
        
        if state.crashed:
            score -= 50
            # Penalize high impact velocity
            impact_v = np.sqrt(state.vx**2 + state.vy**2)
            score -= impact_v * 2
            
        if state.landed:
            score += 200
            score += state.fuel # Bonus for saving fuel
            score += (600 - steps) * 0.1 # Bonus for speed
            
        self.fitness = max(0, score)

# --- 4. THE EVOLUTION ENGINE ---

class GeneticAlgorithm:
    def __init__(self):
        self.population = [Agent() for _ in range(CONFIG['POPULATION_SIZE'])]
        self.generation = 0
        self.best_fitness_history = []
        self.adaptive_rate = CONFIG['MUTATION_RATE']

    def evolve(self):
        # 1. Sort by fitness (Descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        best_agent = self.population[0]
        self.best_fitness_history.append(best_agent.fitness)
        
        # 2. Elitism: Keep the top 20% unchanged
        elite_count = int(CONFIG['POPULATION_SIZE'] * 0.2)
        new_pop = self.population[:elite_count]
        
        # 3. Adaptive Mutation Logic
        # If no improvement in last 5 gens, increase mutation rate (panic mode)
        if len(self.best_fitness_history) > 5:
            recent_avg = np.mean(self.best_fitness_history[-5:])
            if best_agent.fitness <= recent_avg + 1.0:
                self.adaptive_rate = min(0.5, self.adaptive_rate * 1.2)
            else:
                self.adaptive_rate = CONFIG['MUTATION_RATE']

        # 4. Breeding (Crossover + Mutation)
        while len(new_pop) < CONFIG['POPULATION_SIZE']:
            parent_a = self.tournament_select()
            parent_b = self.tournament_select()
            
            child = self.crossover(parent_a, parent_b)
            child.brain.mutate(rate=self.adaptive_rate, intensity=0.5)
            new_pop.append(child)
            
        self.population = new_pop
        self.generation += 1
        return best_agent

    def tournament_select(self, k=3):
        candidates = random.sample(self.population, k)
        return max(candidates, key=lambda x: x.fitness)

    def crossover(self, parent_a, parent_b):
        child = Agent()
        # Layer-wise crossover
        for i in range(len(parent_a.brain.layers)):
            # Randomly inherit whole layer weights from A or B
            if random.random() > 0.5:
                child.brain.layers[i] = copy.deepcopy(parent_a.brain.layers[i])
                child.brain.biases[i] = copy.deepcopy(parent_a.brain.biases[i])
            else:
                child.brain.layers[i] = copy.deepcopy(parent_b.brain.layers[i])
                child.brain.biases[i] = copy.deepcopy(parent_b.brain.biases[i])
        return child

# --- 5. VISUALIZATION DASHBOARD ---

def run_simulation():
    print("Initializing Project Aether...")
    ga = GeneticAlgorithm()
    
    # Setup Visualization
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2)
    
    ax_sim = fig.add_subplot(gs[:, 0])
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_brain = fig.add_subplot(gs[1, 1])
    
    # Static Elements
    ground = patches.Rectangle((-50, -2), 100, 2, color='#444444')
    pad = patches.Rectangle((-2, 0), 4, 0.5, color='#00ff00', alpha=0.5)
    
    lander_body = patches.Circle((0,0), 1.5, color='#3498db')
    thrust_plume = patches.Polygon([[0,0], [-1,-3], [1,-3]], color='orange', alpha=0)
    
    # Data containers for animation
    stats_lines, = ax_stats.plot([], [], 'c-', linewidth=2)
    
    def init():
        ax_sim.set_xlim(-40, 40)
        ax_sim.set_ylim(-5, 60)
        ax_sim.add_patch(ground)
        ax_sim.add_patch(pad)
        ax_sim.add_patch(lander_body)
        ax_sim.add_patch(thrust_plume)
        ax_sim.set_title("Real-time Flight Telemetry")
        ax_sim.set_xlabel("Lateral Distance (m)")
        ax_sim.set_ylabel("Altitude (m)")
        
        ax_stats.set_title("Fitness Evolution")
        ax_stats.set_xlim(0, CONFIG['GENERATIONS'])
        ax_stats.set_ylim(0, 300)
        ax_stats.grid(True, alpha=0.2)
        
        ax_brain.set_title("Neural Activity (Output Layer)")
        ax_brain.axis('off')
        
        return lander_body, thrust_plume, stats_lines

    def update(frame):
        # 1. Run Generation Evaluation
        start_x = random.uniform(-30, 30) # Random spawn x
        start_y = 50                      # Fixed spawn y
        
        for agent in ga.population:
            agent.evaluate(start_x, start_y)
            
        best_agent = ga.evolve()
        
        # 2. Re-run best agent for visualization (deterministic re-play)
        # We simulate the best agent step-by-step for the frame
        
        state = LanderState(start_x, start_y, 0, 0, CONFIG['MAX_FUEL'])
        trajectory_x = []
        trajectory_y = []
        
        # We need to simulate the whole flight instantly to plot trajectory
        # or simulate one frame per animation tick? 
        # For smoothness, we simulate the WHOLE flight of the best agent 
        # and draw the path.
        
        flight_path = []
        
        steps = 0
        while not (state.landed or state.crashed) and steps < 600:
            inputs = np.array([state.x/50, state.y/50, state.vx/10, state.vy/10, state.angle, state.fuel/CONFIG['MAX_FUEL']])
            action = best_agent.brain.forward(inputs)
            thrust = (action[0] + 1) / 2
            rotate = action[1]
            state.update(thrust, rotate, CONFIG['DT'])
            flight_path.append((state.x, state.y))
            steps += 1
            
        # Update Visuals with final state of best agent
        lander_body.center = (state.x, state.y)
        
        # Color code result
        if state.landed:
            lander_body.set_color('#2ecc71') # Green
        elif state.crashed:
            lander_body.set_color('#e74c3c') # Red
        else:
            lander_body.set_color('#3498db') # Blue
            
        # Draw Trajectory
        path_x, path_y = zip(*flight_path) if flight_path else ([], [])
        if len(ax_sim.lines) > 0:
            ax_sim.lines[0].set_data(path_x, path_y)
        else:
            ax_sim.plot(path_x, path_y, 'w--', alpha=0.3, linewidth=1)
            
        # Update Stats
        stats_lines.set_data(range(len(ga.best_fitness_history)), ga.best_fitness_history)
        
        # Print info
        print(f"Gen {ga.generation}: Best Fitness {best_agent.fitness:.2f} | Fuel {state.fuel:.1f} | Result: {'LANDED' if state.landed else 'CRASH'}")
        
        return lander_body, stats_lines

    # Run Animation
    anim = FuncAnimation(fig, update, frames=CONFIG['GENERATIONS'], init_func=init, blit=False, interval=100)
    
    try:
        plt.show()
    except Exception as e:
        print("Visualization skipped (headless environment detected).")
        # Fallback for non-GUI environments: Run in console
        for _ in range(CONFIG['GENERATIONS']):
            update(0)

if __name__ == "__main__":
    run_simulation()