"""
Configuration settings for the dating market simulation.

This module contains all configuration constants for the simulation including
display settings, simulation parameters, ML model paths, and visual styling.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SimulationConfig:
    """Configuration class for simulation parameters."""
    
    # Display settings
    WINDOW_WIDTH: int = 1280
    WINDOW_HEIGHT: int = 720
    FPS: int = 60
    CANVAS_WIDTH: int = 800
    CANVAS_HEIGHT: int = 720
    PANEL_WIDTH: int = 480
    PANEL_HEIGHT: int = 720
    
    # Simulation parameters
    INITIAL_AGENTS: int = 6
    MIN_AGENTS: int = 2
    MAX_AGENTS: int = 10
    AGENT_SPEED: float = 50.0  # pixels/second
    MIN_AGENT_SPEED: float = 25.0  # pixels/second
    MAX_AGENT_SPEED: float = 150.0  # pixels/second
    DEFAULT_AGENT_SPEED: float = 50.0  # pixels/second
    INTERACTION_RADIUS: float = 30.0
    DATING_DURATION: float = 2.0  # seconds (reduced for simulation speed)
    
    # Simulation speed
    MIN_SPEED: float = 0.5
    MAX_SPEED: float = 5.0
    DEFAULT_SPEED: float = 1.0
    
    # Age parameters
    MIN_AGE: int = 18
    MAX_AGE: int = 58
    DEFAULT_MIN_AGE: int = 21
    DEFAULT_MAX_AGE: int = 35
    
    # Match threshold
    MIN_MATCH_THRESHOLD: float = 0.0
    MAX_MATCH_THRESHOLD: float = 1.0
    DEFAULT_MATCH_THRESHOLD: float = 0.5
    
    # Field diversity
    MIN_FIELD_DIVERSITY: int = 0
    MAX_FIELD_DIVERSITY: int = 100
    DEFAULT_FIELD_DIVERSITY: int = 50
    
    # ML Models paths
    RANDOM_FOREST_PATH: str = "decision_tree_results/models/random_forest_model.pkl"
    ASSOCIATION_RULES_PATH: str = "apriori_results/data/association_rules.csv"
    CLEANED_DATA_PATH: str = "Speed_Dating_Data_Cleaned.csv"
    FEATURE_IMPORTANCE_PATH: str = "decision_tree_results/data/feature_importance_random_forest.csv"
    
    # Colors (RGB)
    COLOR_MALE: Tuple[int, int, int] = (100, 150, 255)
    COLOR_FEMALE: Tuple[int, int, int] = (255, 150, 200)
    COLOR_MATCH: Tuple[int, int, int] = (100, 255, 100)
    COLOR_DATING: Tuple[int, int, int] = (255, 215, 0)
    COLOR_BACKGROUND: Tuple[int, int, int] = (240, 240, 245)
    COLOR_PANEL_BG: Tuple[int, int, int] = (255, 255, 255)
    COLOR_TEXT: Tuple[int, int, int] = (50, 50, 50)
    COLOR_TEXT_LIGHT: Tuple[int, int, int] = (100, 100, 100)
    
    # Agent rendering
    MIN_AGENT_RADIUS: int = 8
    MAX_AGENT_RADIUS: int = 15
    AGENT_OUTLINE_WIDTH: int = 2
    
    # UI settings
    FONT_SIZE_LARGE: int = 24
    FONT_SIZE_MEDIUM: int = 18
    FONT_SIZE_SMALL: int = 14
    
    # Metrics update frequency
    METRICS_UPDATE_INTERVAL: int = 30  # frames
    
    # Chart settings
    CHART_WIDTH: int = 450
    CHART_HEIGHT: int = 200
    MAX_DATA_POINTS: int = 100
    
    # Top rules to display
    TOP_RULES_COUNT: int = 5
    
    # Logging
    LOG_LEVEL: str = "INFO"


# Singleton instance
config = SimulationConfig()
