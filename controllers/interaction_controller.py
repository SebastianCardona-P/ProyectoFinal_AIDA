"""
Interaction controller for managing dating encounters.

This module handles the logic for agent interactions and dating encounters.
"""

import logging
import random
from typing import Optional
from models.agent import Agent, AgentState
from models.predictor import match_predictor
from models.rules_engine import rules_engine
from utils.metrics_tracker import MetricsTracker
from config.simulation_config import config

logger = logging.getLogger(__name__)


class InteractionController:
    """Controls dating interactions between agents."""
    
    def __init__(self, metrics_tracker: MetricsTracker):
        """
        Initialize interaction controller.
        
        Args:
            metrics_tracker: Metrics tracker instance.
        """
        self.metrics_tracker = metrics_tracker
        self.match_threshold = config.DEFAULT_MATCH_THRESHOLD
        
        # Current dating interactions with metadata
        self.current_interactions = []  # List of dicts with interaction details
        
    def process_encounter(self, agent1: Agent, agent2: Agent) -> bool:
        """
        Process a dating encounter between two agents.
        
        Args:
            agent1: First agent.
            agent2: Second agent.
            
        Returns:
            True if a match occurred, False otherwise.
        """
        # Verify agents can interact
        if not agent1.can_interact_with(agent2):
            return False
        
        # Start dating
        agent1.start_date(agent2)
        agent2.start_date(agent1)
        
        logger.debug(f"Dating started: Agent {agent1.id} and Agent {agent2.id}")
        
        return True
    
    def evaluate_match(self, agent1: Agent, agent2: Agent) -> bool:
        """
        Evaluate whether a dating encounter results in a match.
        
        Args:
            agent1: First agent.
            agent2: Second agent.
            
        Returns:
            True if matched, False otherwise.
        """
        # Get prediction from Random Forest
        base_probability = match_predictor.predict_match_probability(agent1, agent2)
        
        # Get compatibility boost from association rules
        rules_result = rules_engine.evaluate_rules(agent1, agent2)
        compatibility_boost = rules_result['compatibility_boost']
        active_rules = rules_result.get('active_rules', [])
        
        # Combine probabilities
        final_probability = min(1.0, base_probability + compatibility_boost)
        
        # Store interaction details for UI display
        interaction_data = {
            'agent1': agent1,
            'agent2': agent2,
            'rf_probability': base_probability,
            'apriori_boost': compatibility_boost,
            'active_rules': active_rules,
            'final_probability': final_probability,
            'in_progress': True
        }
        self._update_interaction_display(agent1.id, agent2.id, interaction_data)
        
        # Determine match based on threshold and probability
        matched = final_probability >= self.match_threshold
        
        # Add some randomness for realism
        if not matched and random.random() < 0.1:  # 10% chance of unexpected match
            matched = True
        
        # Prepare agent attributes for history
        agent1_attrs = {
            'gender': 'M' if agent1.gender else 'F',
            'age': agent1.age,
            'race': agent1.race,
            'field': agent1.field,
            'attractiveness': agent1.attractiveness,
            'sincerity': agent1.sincerity,
            'intelligence': agent1.intelligence,
            'fun': agent1.fun,
            'ambition': agent1.ambition
        }
        
        agent2_attrs = {
            'gender': 'M' if agent2.gender else 'F',
            'age': agent2.age,
            'race': agent2.race,
            'field': agent2.field,
            'attractiveness': agent2.attractiveness,
            'sincerity': agent2.sincerity,
            'intelligence': agent2.intelligence,
            'fun': agent2.fun,
            'ambition': agent2.ambition
        }
        
        # Record metrics with full details
        self.metrics_tracker.record_encounter(
            agent1_id=agent1.id,
            agent2_id=agent2.id,
            matched=matched,
            probability=final_probability,
            compatibility_boost=compatibility_boost,
            rf_probability=base_probability,
            active_rules=active_rules,
            agent1_attrs=agent1_attrs,
            agent2_attrs=agent2_attrs
        )
        
        if matched:
            # Record match details
            self.metrics_tracker.record_match_details(
                field1=agent1.field,
                field2=agent2.field,
                race1=agent1.race,
                race2=agent2.race,
                age_gap=abs(agent1.age - agent2.age)
            )
        
        logger.debug(f"Match evaluation: Agent {agent1.id} & {agent2.id} - "
                    f"Base: {base_probability:.2f}, Boost: {compatibility_boost:.2f}, "
                    f"Final: {final_probability:.2f}, Matched: {matched}")
        
        return matched
    
    def end_encounter(self, agent1: Agent, agent2: Agent):
        """
        End a dating encounter.
        
        Args:
            agent1: First agent.
            agent2: Second agent.
        """
        # Evaluate match
        matched = self.evaluate_match(agent1, agent2)
        
        # End date for both agents
        agent1.end_date(matched=matched)
        agent2.end_date(matched=matched)
        
        # Remove from current interactions
        self._remove_interaction_display(agent1.id, agent2.id)
        
        logger.debug(f"Dating ended: Agent {agent1.id} and Agent {agent2.id}, Matched: {matched}")
        
        return matched
    
    def _update_interaction_display(self, agent1_id: int, agent2_id: int, interaction_data: dict):
        """Update current interaction display data."""
        # Remove existing interaction if present
        self.current_interactions = [
            i for i in self.current_interactions 
            if not ((i['agent1'].id == agent1_id and i['agent2'].id == agent2_id) or
                    (i['agent1'].id == agent2_id and i['agent2'].id == agent1_id))
        ]
        # Add new interaction
        self.current_interactions.append(interaction_data)
    
    def _remove_interaction_display(self, agent1_id: int, agent2_id: int):
        """Remove interaction from display."""
        self.current_interactions = [
            i for i in self.current_interactions 
            if not ((i['agent1'].id == agent1_id and i['agent2'].id == agent2_id) or
                    (i['agent1'].id == agent2_id and i['agent2'].id == agent1_id))
        ]
    
    def get_current_interactions(self):
        """Get list of current dating interactions with metadata."""
        return self.current_interactions
    
    def update_dating_timers(self, agents, delta_time: float) -> int:
        """
        Update timers for all dating agents.
        
        Args:
            agents: List of all agents.
            delta_time: Time elapsed since last update.
            
        Returns:
            Number of encounters that ended.
        """
        encounters_ended = 0
        
        # Track which agents to end dating for
        agents_to_end = []
        
        for agent in agents:
            if agent.state == AgentState.DATING and agent.current_date_partner:
                agent.date_timer += delta_time
                
                # Check if dating duration exceeded
                if agent.date_timer >= config.DATING_DURATION:
                    # Only process once per pair
                    if agent.id < agent.current_date_partner.id:
                        agents_to_end.append((agent, agent.current_date_partner))
        
        # End encounters
        for agent1, agent2 in agents_to_end:
            self.end_encounter(agent1, agent2)
            encounters_ended += 1
        
        return encounters_ended
    
    def set_match_threshold(self, threshold: float):
        """
        Set the match probability threshold.
        
        Args:
            threshold: New threshold (0.0 to 1.0).
        """
        self.match_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Match threshold set to {self.match_threshold:.2f}")
