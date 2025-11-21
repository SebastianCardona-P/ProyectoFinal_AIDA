"""
Match predictor using Random Forest model.

This module provides an interface for predicting match probability between agents.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from utils.data_loader import model_loader

logger = logging.getLogger(__name__)


class MatchPredictor:
    """Predictor for match probability using Random Forest."""
    
    def __init__(self):
        """Initialize the match predictor."""
        self.model = None
        self.feature_names = None
        self._prediction_cache: Dict[tuple, float] = {}
        
    def initialize(self):
        """Load the Random Forest model."""
        if self.model is None:
            logger.info("Initializing Match Predictor")
            self.model = model_loader.load_random_forest()
            
            # Get feature names from model if available
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
            else:
                # Load from feature importance file
                feature_importance = model_loader.load_feature_importance()
                if not feature_importance.empty:
                    self.feature_names = feature_importance['feature'].tolist()
            
            logger.info(f"Match Predictor initialized with {len(self.feature_names) if self.feature_names else 0} features")
    
    def prepare_feature_vector(self, agent1: 'Agent', agent2: 'Agent') -> Optional[np.ndarray]:
        """
        Prepare feature vector from two agents.
        
        Args:
            agent1: First agent.
            agent2: Second agent.
            
        Returns:
            Feature vector as numpy array, or None if features can't be prepared.
        """
        if self.feature_names is None:
            logger.warning("Feature names not available")
            return None
        
        try:
            features = {}
            
            # Basic attributes
            features['attr'] = agent1.attractiveness
            features['sinc'] = agent1.sincerity
            features['intel'] = agent1.intelligence
            features['fun'] = agent1.fun
            features['amb'] = agent1.ambition
            features['shar'] = agent1.shared_interests
            
            # Partner attributes (what agent2 has)
            features['attr_o'] = agent2.attractiveness
            features['sinc_o'] = agent2.sincerity
            features['intel_o'] = agent2.intelligence
            features['fun_o'] = agent2.fun
            features['amb_o'] = agent2.ambition
            features['shar_o'] = agent2.shared_interests
            
            # Age
            features['age'] = agent1.age
            features['age_o'] = agent2.age
            features['age_diff'] = abs(agent1.age - agent2.age)
            
            # Age gap category (encoded)
            age_diff = abs(agent1.age - agent2.age)
            features['age_gap_category'] = 0 if age_diff <= 2 else (1 if age_diff <= 5 else 2)
            
            # Race
            features['race_o'] = agent2.race
            features['samerace'] = 1.0 if agent1.race == agent2.race else 0.0
            
            # Field
            features['field'] = agent1.field
            
            # Differences
            features['attr_diff'] = abs(agent1.attractiveness - agent2.attractiveness)
            features['sinc_diff'] = abs(agent1.sincerity - agent2.sincerity)
            features['intel_diff'] = abs(agent1.intelligence - agent2.intelligence)
            features['fun_diff'] = abs(agent1.fun - agent2.fun)
            features['amb_diff'] = abs(agent1.ambition - agent2.ambition)
            features['shar_diff'] = abs(agent1.shared_interests - agent2.shared_interests)
            
            # Average ratings
            agent1_avg = np.mean([agent1.attractiveness, agent1.sincerity, agent1.intelligence, 
                                 agent1.fun, agent1.ambition, agent1.shared_interests])
            agent2_avg = np.mean([agent2.attractiveness, agent2.sincerity, agent2.intelligence,
                                 agent2.fun, agent2.ambition, agent2.shared_interests])
            
            features['avg_rating_given'] = agent1_avg
            features['avg_rating_received'] = agent2_avg
            features['rating_asymmetry'] = agent1_avg - agent2_avg
            
            # Preference match score (simplified)
            pref_match = 100 - (features['attr_diff'] + features['sinc_diff'] + 
                               features['intel_diff'] + features['fun_diff'] + 
                               features['amb_diff'] + features['shar_diff']) * 2
            features['preference_match_score'] = max(0, min(100, pref_match))
            
            # Partner preferences (default values)
            features['pf_o_att'] = 3.0
            features['pf_o_sin'] = 2.0
            features['pf_o_int'] = 2.5
            features['pf_o_fun'] = 2.5
            features['pf_o_amb'] = 1.5
            features['pf_o_sha'] = 1.5
            
            # Create feature vector in correct order
            feature_vector = []
            for fname in self.feature_names:
                if fname in features:
                    feature_vector.append(features[fname])
                elif fname.startswith('race_'):
                    # Binary encoding for race
                    race_num = fname.split('_')[1]
                    feature_vector.append(1.0 if str(int(agent1.race)) == race_num else 0.0)
                elif fname.startswith('race_o_'):
                    # Binary encoding for partner race
                    race_num = fname.split('_')[2]
                    feature_vector.append(1.0 if str(int(agent2.race)) == race_num else 0.0)
                elif fname.startswith('field_'):
                    # Binary encoding for field
                    field_num = fname.split('_')[1]
                    feature_vector.append(1.0 if str(int(agent1.field)) == field_num else 0.0)
                else:
                    # Default value for missing features
                    feature_vector.append(0.0)
            
            return np.array(feature_vector).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing feature vector: {e}")
            return None
    
    def predict_match_probability(self, agent1: 'Agent', agent2: 'Agent') -> float:
        """
        Predict match probability between two agents.
        
        Args:
            agent1: First agent.
            agent2: Second agent.
            
        Returns:
            Probability of match (0.0 to 1.0).
        """
        if self.model is None:
            self.initialize()
        
        # Check cache
        cache_key = (agent1.id, agent2.id)
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
        
        # Prepare features
        features = self.prepare_feature_vector(agent1, agent2)
        if features is None:
            logger.warning(f"Could not prepare features for agents {agent1.id} and {agent2.id}")
            return 0.5  # Default probability
        
        try:
            # Predict probability
            if hasattr(self.model, 'predict_proba'):
                # Get probability of positive class (match)
                proba = self.model.predict_proba(features)[0, 1]
            else:
                # Fallback to prediction
                pred = self.model.predict(features)[0]
                proba = float(pred)
            
            # Cache result
            self._prediction_cache[cache_key] = proba
            self._prediction_cache[(agent2.id, agent1.id)] = proba  # Symmetric
            
            return float(proba)
            
        except Exception as e:
            logger.error(f"Error predicting match probability: {e}")
            return 0.5
    
    def clear_cache(self):
        """Clear the prediction cache."""
        self._prediction_cache.clear()


# Singleton instance
match_predictor = MatchPredictor()
