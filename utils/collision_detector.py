"""
Spatial collision detection using grid-based partitioning.

This module provides efficient O(n) collision detection for agent interactions.
"""

import logging
from typing import List, Dict, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class CollisionDetector:
    """Grid-based spatial hash for efficient collision detection."""
    
    def __init__(self, cell_size: float = 50.0):
        """
        Initialize the collision detector.
        
        Args:
            cell_size: Size of each grid cell for spatial hashing.
        """
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], List] = defaultdict(list)
        
    def clear(self):
        """Clear the spatial grid."""
        self.grid.clear()
    
    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        """
        Get grid cell coordinates for a position.
        
        Args:
            x: X coordinate.
            y: Y coordinate.
            
        Returns:
            Cell coordinates (cell_x, cell_y).
        """
        cell_x = int(x // self.cell_size)
        cell_y = int(y // self.cell_size)
        return (cell_x, cell_y)
    
    def insert(self, agent: 'Agent'):
        """
        Insert an agent into the spatial grid.
        
        Args:
            agent: Agent to insert.
        """
        cell = self._get_cell(agent.position[0], agent.position[1])
        self.grid[cell].append(agent)
    
    def get_nearby_agents(self, agent: 'Agent', radius: float) -> List['Agent']:
        """
        Get all agents within a radius of the given agent.
        
        Args:
            agent: Center agent.
            radius: Search radius.
            
        Returns:
            List of nearby agents (excluding the agent itself).
        """
        nearby = []
        
        # Get agent's cell
        agent_cell = self._get_cell(agent.position[0], agent.position[1])
        
        # Check neighboring cells (3x3 grid around agent)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell = (agent_cell[0] + dx, agent_cell[1] + dy)
                
                if cell in self.grid:
                    for other in self.grid[cell]:
                        if other.id != agent.id:
                            # Check distance
                            dist_sq = (
                                (agent.position[0] - other.position[0]) ** 2 +
                                (agent.position[1] - other.position[1]) ** 2
                            )
                            
                            if dist_sq <= radius ** 2:
                                nearby.append(other)
        
        return nearby
    
    def build_grid(self, agents: List['Agent']):
        """
        Build the spatial grid from a list of agents.
        
        Args:
            agents: List of all agents.
        """
        self.clear()
        for agent in agents:
            self.insert(agent)
