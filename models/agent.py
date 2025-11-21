"""
Agent model for the dating market simulation.

This module defines the Agent class and AgentFactory for creating agents.
"""

import logging
from enum import Enum
from typing import Tuple, List, Optional
import numpy as np
from utils.data_loader import data_stats

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Enum for agent states."""
    IDLE = "idle"
    DATING = "dating"
    MATCHED = "matched"


class Agent:
    """Represents a participant in the speed dating simulation."""
    
    _id_counter = 0
    
    def __init__(self, 
                 gender: bool,
                 age: int,
                 race: int,
                 field: int,
                 attractiveness: float,
                 sincerity: float,
                 intelligence: float,
                 fun: float,
                 ambition: float,
                 shared_interests: float,
                 position: Tuple[float, float] = (0, 0)):
        """
        Initialize an agent.
        
        Args:
            gender: True for male, False for female.
            age: Age of the agent.
            race: Race code (0-6).
            field: Field/career code (1-18).
            attractiveness: Attractiveness rating (0-10).
            sincerity: Sincerity rating (0-10).
            intelligence: Intelligence rating (0-10).
            fun: Fun rating (0-10).
            ambition: Ambition rating (0-10).
            shared_interests: Shared interests rating (0-10).
            position: Initial position (x, y).
        """
        Agent._id_counter += 1
        self.id = Agent._id_counter
        
        # Demographics
        self.gender = gender  # True = Male, False = Female
        self.age = age
        self.race = race
        self.field = field
        
        # Attributes
        self.attractiveness = attractiveness
        self.sincerity = sincerity
        self.intelligence = intelligence
        self.fun = fun
        self.ambition = ambition
        self.shared_interests = shared_interests
        
        # Simulation properties
        self.position = list(position)  # [x, y]
        self.velocity = [
            np.random.uniform(-1, 1),
            np.random.uniform(-1, 1)
        ]
        self.state = AgentState.IDLE
        
        # Interaction properties
        self.current_date_partner: Optional['Agent'] = None
        self.date_timer: float = 0.0
        self.interest_level: float = 7.0  # General interest in dating
        
        # History
        self.encounters: List[int] = []  # Agent IDs encountered
        self.matches: List[int] = []  # Agent IDs matched with
        
    def update_position(self, delta_time: float, boundaries: Tuple[int, int], speed: float):
        """
        Update agent position based on velocity.
        
        Args:
            delta_time: Time elapsed since last update (seconds).
            boundaries: (width, height) of the canvas.
            speed: Movement speed multiplier.
        """
        if self.state == AgentState.DATING:
            # Don't move while dating
            return
        
        # Update position
        self.position[0] += self.velocity[0] * speed * delta_time
        self.position[1] += self.velocity[1] * speed * delta_time
        
        # Bounce off boundaries
        if self.position[0] <= 0 or self.position[0] >= boundaries[0]:
            self.velocity[0] *= -1
            self.position[0] = max(0, min(boundaries[0], self.position[0]))
        
        if self.position[1] <= 0 or self.position[1] >= boundaries[1]:
            self.velocity[1] *= -1
            self.position[1] = max(0, min(boundaries[1], self.position[1]))
    
    def can_interact_with(self, other: 'Agent') -> bool:
        """
        Check if this agent can interact with another agent.
        
        Args:
            other: Another agent.
            
        Returns:
            True if interaction is possible, False otherwise.
        """
        # Can't interact with self
        if self.id == other.id:
            return False
        
        # Must be opposite genders
        if self.gender == other.gender:
            return False
        
        # Can't be currently dating someone else
        if self.state == AgentState.DATING or other.state == AgentState.DATING:
            return False
        
        # Haven't met before
        if other.id in self.encounters:
            return False
        
        return True
    
    def start_date(self, partner: 'Agent'):
        """
        Start a date with a partner.
        
        Args:
            partner: The dating partner.
        """
        self.state = AgentState.DATING
        self.current_date_partner = partner
        self.date_timer = 0.0
    
    def end_date(self, matched: bool = False):
        """
        End the current date.
        
        Args:
            matched: Whether the date resulted in a match.
        """
        if self.current_date_partner:
            self.encounters.append(self.current_date_partner.id)
            
            if matched:
                self.matches.append(self.current_date_partner.id)
                self.state = AgentState.MATCHED
            else:
                self.state = AgentState.IDLE
            
            self.current_date_partner = None
            self.date_timer = 0.0
    
    def reset_state(self):
        """Reset agent to idle state."""
        if self.state == AgentState.MATCHED:
            # Keep matched state
            return
        self.state = AgentState.IDLE
        self.current_date_partner = None
        self.date_timer = 0.0
    
    def get_gender_str(self) -> str:
        """Get gender as string."""
        return "Male" if self.gender else "Female"
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"Agent({self.id}, {self.get_gender_str()}, age={self.age})"


class AgentFactory:
    """Factory for creating agents with realistic attributes."""
    
    def __init__(self):
        """Initialize the agent factory."""
        self.stats = None
        
    def initialize(self):
        """Load data statistics."""
        if self.stats is None:
            logger.info("Initializing Agent Factory")
            self.stats = data_stats.compute_statistics()
            logger.info("Agent Factory initialized")
    
    def generate_agents(self, 
                       num_agents: int,
                       min_age: int = 21,
                       max_age: int = 35,
                       field_diversity: float = 0.5) -> List[Agent]:
        """
        Generate agents with realistic attributes.
        
        Args:
            num_agents: Number of agents to generate.
            min_age: Minimum age.
            max_age: Maximum age.
            field_diversity: Field diversity (0.0 to 1.0).
            
        Returns:
            List of generated agents.
        """
        if self.stats is None:
            self.initialize()
        
        logger.info(f"Generating {num_agents} agents")
        agents = []
        
        # Ensure gender balance
        num_male = num_agents // 2
        num_female = num_agents - num_male
        
        # Generate male agents
        for _ in range(num_male):
            agent = self._create_agent(
                gender=True,
                min_age=min_age,
                max_age=max_age,
                field_diversity=field_diversity
            )
            agents.append(agent)
        
        # Generate female agents
        for _ in range(num_female):
            agent = self._create_agent(
                gender=False,
                min_age=min_age,
                max_age=max_age,
                field_diversity=field_diversity
            )
            agents.append(agent)
        
        logger.info(f"Generated {len(agents)} agents ({num_male} male, {num_female} female)")
        return agents
    
    def _create_agent(self, 
                     gender: bool,
                     min_age: int,
                     max_age: int,
                     field_diversity: float) -> Agent:
        """
        Create a single agent.
        
        Args:
            gender: True for male, False for female.
            min_age: Minimum age.
            max_age: Maximum age.
            field_diversity: Field diversity.
            
        Returns:
            Created agent.
        """
        # Sample age
        if 'age' in self.stats:
            age = int(np.random.normal(self.stats['age']['mean'], self.stats['age']['std']))
            age = max(min_age, min(max_age, age))
        else:
            age = np.random.randint(min_age, max_age + 1)
        
        # Sample race
        race = self._sample_categorical('race_distribution', default=2)
        
        # Sample field with diversity control
        if np.random.random() < field_diversity:
            field = self._sample_categorical('field_distribution', default=1)
        else:
            # Concentrate on popular fields
            field = np.random.choice([1, 2, 3, 4, 5])  # Law, econ, etc.
        
        # Sample attributes
        attractiveness = self._sample_attribute('attr')
        sincerity = self._sample_attribute('sinc')
        intelligence = self._sample_attribute('intel')
        fun = self._sample_attribute('fun')
        ambition = self._sample_attribute('amb')
        shared_interests = self._sample_attribute('shar')
        
        # Random initial position (will be set by controller)
        position = (0, 0)
        
        return Agent(
            gender=gender,
            age=age,
            race=race,
            field=field,
            attractiveness=attractiveness,
            sincerity=sincerity,
            intelligence=intelligence,
            fun=fun,
            ambition=ambition,
            shared_interests=shared_interests,
            position=position
        )
    
    def _sample_categorical(self, dist_key: str, default: int) -> int:
        """Sample from a categorical distribution."""
        if dist_key in self.stats:
            dist = self.stats[dist_key]
            categories = list(dist.keys())
            probabilities = list(dist.values())
            
            # Normalize probabilities
            total = sum(probabilities)
            if total > 0:
                probabilities = [p / total for p in probabilities]
                choice = np.random.choice(categories, p=probabilities)
                return int(float(choice))
        
        return default
    
    def _sample_attribute(self, attr_name: str) -> float:
        """Sample an attribute value."""
        if attr_name in self.stats:
            stat = self.stats[attr_name]
            value = np.random.normal(stat['mean'], stat['std'])
            value = np.clip(value, stat.get('min', 0), stat.get('max', 10))
            return float(value)
        
        # Default to mid-range with some variance
        return float(np.clip(np.random.normal(6.0, 2.0), 0, 10))


# Singleton instance
agent_factory = AgentFactory()
