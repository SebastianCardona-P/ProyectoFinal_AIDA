"""
Association rules engine for compatibility boost.

This module evaluates association rules to enhance match predictions.
"""

import logging
from typing import Dict, List, Tuple, Any
import pandas as pd
from utils.data_loader import rules_loader

logger = logging.getLogger(__name__)


class AssociationRulesEngine:
    """Engine for evaluating association rules between agents."""
    
    def __init__(self):
        """Initialize the rules engine."""
        self.rules = None
        self.top_rules = None
        
    def initialize(self):
        """Load association rules."""
        if self.rules is None:
            logger.info("Initializing Association Rules Engine")
            self.rules = rules_loader.load_association_rules(max_rows=500)
            if not self.rules.empty:
                self.top_rules = rules_loader.get_top_rules(n=20, min_confidence=0.5)
                logger.info(f"Loaded {len(self.rules)} rules, {len(self.top_rules)} top rules")
            else:
                logger.warning("No association rules loaded")
                self.top_rules = pd.DataFrame()
    
    def evaluate_rules(self, agent1: 'Agent', agent2: 'Agent') -> Dict[str, float]:
        """
        Evaluate association rules for two agents.
        
        Args:
            agent1: First agent.
            agent2: Second agent.
            
        Returns:
            Dictionary with rule evaluation results:
                - compatibility_boost: Boost value (0.0 to 1.0)
                - active_rules: List of active rules
                - rule_scores: Individual rule scores
        """
        if self.rules is None or self.rules.empty:
            self.initialize()
        
        if self.top_rules is None or self.top_rules.empty:
            return {
                'compatibility_boost': 0.0,
                'active_rules': [],
                'rule_scores': {}
            }
        
        active_rules = []
        rule_scores = {}
        total_boost = 0.0
        
        try:
            # Check if agents meet conditions for top rules
            # Simplified rule matching based on attribute values
            
            # Rule 1: High attractiveness received + Said Yes → High fun + Match
            if agent2.attractiveness >= 7.0 and agent1.interest_level >= 7.0:
                boost = 0.15
                total_boost += boost
                active_rules.append("High Attraction + Interest → Match")
                rule_scores["attr_interest"] = boost
            
            # Rule 2: High fun + High attractiveness → Match likely
            if agent1.fun >= 7.0 and agent2.attractiveness >= 7.0:
                boost = 0.12
                total_boost += boost
                active_rules.append("High Fun + Attractive → Match")
                rule_scores["fun_attr"] = boost
            
            # Rule 3: Similar interests → Match boost
            if abs(agent1.shared_interests - agent2.shared_interests) <= 2.0:
                boost = 0.10
                total_boost += boost
                active_rules.append("Shared Interests → Match")
                rule_scores["shared_interests"] = boost
            
            # Rule 4: Both intelligent → Match
            if agent1.intelligence >= 7.0 and agent2.intelligence >= 7.0:
                boost = 0.08
                total_boost += boost
                active_rules.append("Mutual Intelligence → Match")
                rule_scores["intelligence"] = boost
            
            # Rule 5: Same race preference
            if agent1.race == agent2.race:
                boost = 0.05
                total_boost += boost
                active_rules.append("Same Race → Match")
                rule_scores["same_race"] = boost
            
            # Rule 6: Age compatibility
            age_diff = abs(agent1.age - agent2.age)
            if age_diff <= 3:
                boost = 0.08
                total_boost += boost
                active_rules.append("Similar Age → Match")
                rule_scores["age_compat"] = boost
            
            # Rule 7: Ambition match
            if abs(agent1.ambition - agent2.ambition) <= 2.0:
                boost = 0.07
                total_boost += boost
                active_rules.append("Similar Ambition → Match")
                rule_scores["ambition"] = boost
            
            # Cap boost at reasonable level
            total_boost = min(total_boost, 0.4)
            
        except Exception as e:
            logger.error(f"Error evaluating rules: {e}")
            total_boost = 0.0
            active_rules = []
            rule_scores = {}
        
        return {
            'compatibility_boost': total_boost,
            'active_rules': active_rules,
            'rule_scores': rule_scores
        }
    
    def get_top_rules_display(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top rules for display in UI.
        
        Args:
            n: Number of rules to return.
            
        Returns:
            List of rule dictionaries.
        """
        if self.top_rules is None or self.top_rules.empty:
            self.initialize()
        
        if self.top_rules is None or self.top_rules.empty:
            return []
        
        display_rules = []
        for idx, row in self.top_rules.head(n).iterrows():
            rule_dict = {
                'antecedents': str(row.get('antecedents', '')),
                'consequents': str(row.get('consequents', '')),
                'confidence': row.get('confidence', 0.0),
                'lift': row.get('lift', 0.0),
                'support': row.get('support', 0.0)
            }
            display_rules.append(rule_dict)
        
        return display_rules


# Singleton instance
rules_engine = AssociationRulesEngine()
