"""
Main simulation controller.

This module orchestrates the entire simulation including agents, interactions, and state.
"""

import logging
import random
from typing import List, Optional
from models.agent import Agent, agent_factory
from models.predictor import match_predictor
from models.rules_engine import rules_engine
from controllers.interaction_controller import InteractionController
from utils.collision_detector import CollisionDetector
from utils.metrics_tracker import MetricsTracker
from utils.data_loader import data_stats
from config.simulation_config import config

logger = logging.getLogger(__name__)


class SimulationController:
    """Main controller for the dating market simulation."""
    
    def __init__(self):
        """Initialize the simulation controller."""
        self.agents: List[Agent] = []
        self.collision_detector = CollisionDetector(cell_size=config.INTERACTION_RADIUS * 2)
        self.metrics_tracker = MetricsTracker()
        self.interaction_controller = InteractionController(self.metrics_tracker)
        
        # State
        self.running = False
        self.paused = False
        self.speed = config.DEFAULT_SPEED
        self.agent_speed = config.DEFAULT_AGENT_SPEED
        
        # Parameters
        self.num_agents = config.INITIAL_AGENTS
        self.min_age = config.DEFAULT_MIN_AGE
        self.max_age = config.DEFAULT_MAX_AGE
        self.field_diversity = config.DEFAULT_FIELD_DIVERSITY / 100.0
        
        # Initialize ML models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models and data."""
        logger.info("Initializing ML models and data...")
        
        # Load data statistics
        data_stats.compute_statistics()
        
        # Initialize agent factory
        agent_factory.initialize()
        
        # Initialize predictor
        match_predictor.initialize()
        
        # Initialize rules engine
        rules_engine.initialize()
        
        logger.info("ML models initialized successfully")
    
    def initialize_simulation(self):
        """Initialize the simulation with agents."""
        logger.info(f"Initializing simulation with {self.num_agents} agents")
        
        # Reset metrics
        self.metrics_tracker.reset()
        
        # Clear prediction cache
        match_predictor.clear_cache()
        
        # Reset agent ID counter
        Agent._id_counter = 0
        
        # Generate agents
        self.agents = agent_factory.generate_agents(
            num_agents=self.num_agents,
            min_age=self.min_age,
            max_age=self.max_age,
            field_diversity=self.field_diversity
        )
        
        # Set random positions within canvas
        for agent in self.agents:
            agent.position = [
                random.uniform(50, config.CANVAS_WIDTH - 50),
                random.uniform(50, config.CANVAS_HEIGHT - 50)
            ]
            # Random velocity
            angle = random.uniform(0, 2 * 3.14159)
            agent.velocity = [
                self.agent_speed * 0.5 * (0.5 + random.random()) * (1 if random.random() > 0.5 else -1),
                self.agent_speed * 0.5 * (0.5 + random.random()) * (1 if random.random() > 0.5 else -1)
            ]
        
        self.running = True
        logger.info("Simulation initialized")
    
    def update(self, delta_time: float):
        """
        Update simulation state.
        
        Args:
            delta_time: Time elapsed since last update (seconds).
        """
        if not self.running or self.paused:
            return
        
        # Apply speed multiplier
        dt = delta_time * self.speed
        
        # Update agent positions
        boundaries = (config.CANVAS_WIDTH, config.CANVAS_HEIGHT)
        for agent in self.agents:
            agent.update_position(dt, boundaries, self.agent_speed)
        
        # Update dating timers
        self.interaction_controller.update_dating_timers(self.agents, dt)
        
        # Build spatial grid for collision detection
        self.collision_detector.build_grid(self.agents)
        
        # Check for new interactions
        self._check_interactions()
    
    def _check_interactions(self):
        """Check for and process new interactions between agents."""
        for agent in self.agents:
            # Skip if already dating or matched
            if agent.state != agent.state.IDLE:
                continue
            
            # Find nearby agents
            nearby = self.collision_detector.get_nearby_agents(
                agent,
                radius=config.INTERACTION_RADIUS
            )
            
            # Try to start dating with first compatible agent
            for other in nearby:
                if agent.can_interact_with(other):
                    # Start encounter
                    success = self.interaction_controller.process_encounter(agent, other)
                    if success:
                        break  # Only one interaction per update
    
    def reset(self):
        """Reset the simulation."""
        logger.info("Resetting simulation")
        self.running = False
        self.paused = False
        self.agents.clear()
        self.metrics_tracker.reset()
        match_predictor.clear_cache()
    
    def set_num_agents(self, num: int):
        """Set number of agents and reinitialize."""
        self.num_agents = max(config.MIN_AGENTS, min(config.MAX_AGENTS, num))
        logger.info(f"Number of agents set to {self.num_agents}")
    
    def set_age_range(self, min_age: int, max_age: int):
        """Set age range for agents."""
        self.min_age = max(config.MIN_AGE, min_age)
        self.max_age = min(config.MAX_AGE, max_age)
        logger.info(f"Age range set to {self.min_age}-{self.max_age}")
    
    def set_field_diversity(self, diversity: float):
        """
        Set field diversity.
        
        Args:
            diversity: Diversity value (0.0 to 1.0).
        """
        self.field_diversity = max(0.0, min(1.0, diversity))
        logger.info(f"Field diversity set to {self.field_diversity:.2f}")
    
    def set_match_threshold(self, threshold: float):
        """
        Set match probability threshold.
        
        Args:
            threshold: Threshold value (0.0 to 1.0).
        """
        self.interaction_controller.set_match_threshold(threshold)
    
    def set_speed(self, speed: float):
        """
        Set simulation speed.
        
        Args:
            speed: Speed multiplier.
        """
        self.speed = max(config.MIN_SPEED, min(config.MAX_SPEED, speed))
        logger.info(f"Simulation speed set to {self.speed}x")
    
    def set_agent_speed(self, agent_speed: float):
        """
        Set agent movement speed.
        
        Args:
            agent_speed: Agent speed in pixels/second.
        """
        self.agent_speed = max(config.MIN_AGENT_SPEED, min(config.MAX_AGENT_SPEED, agent_speed))
        # Update velocity for all existing agents
        for agent in self.agents:
            # Maintain direction, update magnitude
            import math
            current_magnitude = math.sqrt(agent.velocity[0]**2 + agent.velocity[1]**2)
            if current_magnitude > 0:
                scale = (self.agent_speed * 0.5) / current_magnitude
                agent.velocity = [agent.velocity[0] * scale, agent.velocity[1] * scale]
        logger.info(f"Agent speed set to {self.agent_speed:.0f} px/s")
    
    def toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
        logger.info(f"Simulation {'paused' if self.paused else 'resumed'}")
    
    def get_statistics(self):
        """Get current simulation statistics."""
        stats = self.metrics_tracker.get_statistics()
        
        # Add agent counts
        male_count = sum(1 for a in self.agents if a.gender)
        female_count = len(self.agents) - male_count
        matched_count = sum(1 for a in self.agents if len(a.matches) > 0)
        dating_count = sum(1 for a in self.agents if a.state == a.state.DATING)
        
        stats.update({
            'total_agents': len(self.agents),
            'male_agents': male_count,
            'female_agents': female_count,
            'matched_agents': matched_count,
            'currently_dating': dating_count,
            'gender_ratio': male_count / len(self.agents) if self.agents else 0.5
        })
        
        return stats
