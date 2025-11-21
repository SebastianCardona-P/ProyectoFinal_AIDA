"""
Agent renderer for drawing agents on the canvas.

This module handles the visual representation of agents.
"""

import pygame
import math
from models.agent import Agent, AgentState
from config.simulation_config import config


class AgentRenderer:
    """Renderer for drawing agents."""
    
    def __init__(self, surface: pygame.Surface):
        """
        Initialize agent renderer.
        
        Args:
            surface: Pygame surface to draw on.
        """
        self.surface = surface
        
    def draw_agent(self, agent: Agent):
        """
        Draw a single agent.
        
        Args:
            agent: Agent to draw.
        """
        # Determine color based on state and gender
        if agent.state == AgentState.MATCHED:
            color = config.COLOR_MATCH
        elif agent.state == AgentState.DATING:
            color = config.COLOR_DATING
        elif agent.gender:
            color = config.COLOR_MALE
        else:
            color = config.COLOR_FEMALE
        
        # Determine size based on attractiveness
        size_range = config.MAX_AGENT_RADIUS - config.MIN_AGENT_RADIUS
        radius = config.MIN_AGENT_RADIUS + int(
            (agent.attractiveness / 10.0) * size_range
        )
        
        # Draw agent circle
        pos = (int(agent.position[0]), int(agent.position[1]))
        pygame.draw.circle(self.surface, color, pos, radius)
        
        # Draw outline
        outline_color = (0, 0, 0) if agent.state == AgentState.IDLE else (255, 255, 255)
        pygame.draw.circle(self.surface, outline_color, pos, radius, config.AGENT_OUTLINE_WIDTH)
        
        # Draw glow effect for matched agents
        if agent.state == AgentState.MATCHED:
            glow_radius = radius + 5
            glow_color = (*config.COLOR_MATCH, 100)  # With alpha
            for i in range(3):
                pygame.draw.circle(
                    self.surface,
                    config.COLOR_MATCH,
                    pos,
                    glow_radius - i * 2,
                    1
                )
    
    def draw_all_agents(self, agents):
        """
        Draw all agents.
        
        Args:
            agents: List of agents to draw.
        """
        for agent in agents:
            self.draw_agent(agent)
    
    def draw_connection(self, agent1: Agent, agent2: Agent, color=(255, 215, 0), width=2):
        """
        Draw a connection line between two agents.
        
        Args:
            agent1: First agent.
            agent2: Second agent.
            color: Line color.
            width: Line width.
        """
        pos1 = (int(agent1.position[0]), int(agent1.position[1]))
        pos2 = (int(agent2.position[0]), int(agent2.position[1]))
        pygame.draw.line(self.surface, color, pos1, pos2, width)
    
    def draw_dating_connections(self, agents):
        """
        Draw connections for agents currently dating.
        
        Args:
            agents: List of all agents.
        """
        drawn_pairs = set()
        
        for agent in agents:
            if agent.state == AgentState.DATING and agent.current_date_partner:
                pair_id = tuple(sorted([agent.id, agent.current_date_partner.id]))
                
                if pair_id not in drawn_pairs:
                    self.draw_connection(agent, agent.current_date_partner)
                    drawn_pairs.add(pair_id)
