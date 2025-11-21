"""
Main view for the simulation window.

This module coordinates all visual components.
"""

import pygame
from views.agent_renderer import AgentRenderer
from views.ui_panel import UIControlPanel
from config.simulation_config import config


class MainView:
    """Main view coordinating all visual components."""
    
    def __init__(self):
        """Initialize the main view."""
        # Initialize Pygame
        pygame.init()
        
        # Create window
        self.screen = pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        )
        pygame.display.set_caption("Dating Market Simulation")
        
        # Create surfaces
        self.canvas = pygame.Surface((config.CANVAS_WIDTH, config.CANVAS_HEIGHT))
        
        # Create UI panel
        panel_rect = pygame.Rect(
            config.CANVAS_WIDTH,
            0,
            config.PANEL_WIDTH,
            config.PANEL_HEIGHT
        )
        self.ui_panel = UIControlPanel(panel_rect, self.screen, config)
        
        # Create agent renderer
        self.agent_renderer = AgentRenderer(self.canvas)
        
        # Font for labels
        self.font = pygame.font.Font(None, config.FONT_SIZE_SMALL)
        self.font_large = pygame.font.Font(None, config.FONT_SIZE_LARGE)
        
        # Clock for FPS
        self.clock = pygame.time.Clock()
        
    def render(self, agents, paused: bool = False):
        """
        Render the entire view.
        
        Args:
            agents: List of agents to render.
            paused: Whether simulation is paused.
        """
        # Clear screen
        self.screen.fill(config.COLOR_PANEL_BG)
        
        # Clear canvas
        self.canvas.fill(config.COLOR_BACKGROUND)
        
        # Draw agents
        self.agent_renderer.draw_all_agents(agents)
        
        # Draw dating connections
        self.agent_renderer.draw_dating_connections(agents)
        
        # Draw pause overlay if paused
        if paused:
            overlay = pygame.Surface((config.CANVAS_WIDTH, config.CANVAS_HEIGHT))
            overlay.set_alpha(100)
            overlay.fill((50, 50, 50))
            self.canvas.blit(overlay, (0, 0))
            
            # Draw "PAUSED" text
            text = self.font_large.render("PAUSED", True, (255, 255, 255))
            text_rect = text.get_rect(center=(config.CANVAS_WIDTH // 2, config.CANVAS_HEIGHT // 2))
            self.canvas.blit(text, text_rect)
        
        # Blit canvas to screen
        self.screen.blit(self.canvas, (0, 0))
        
        # Draw UI panel
        self.ui_panel.draw()
        
        # Update display
        pygame.display.flip()
    
    def update_ui(self, time_delta: float):
        """
        Update UI (no-op for native pygame UI).
        
        Args:
            time_delta: Time elapsed since last update.
        """
        pass
    
    def process_events(self):
        """
        Process pygame events.
        
        Returns:
            List of pygame events.
        """
        events = pygame.event.get()
        return events
    
    def tick(self, fps: int = config.FPS) -> float:
        """
        Tick the clock and return delta time.
        
        Args:
            fps: Target FPS.
            
        Returns:
            Delta time in seconds.
        """
        return self.clock.tick(fps) / 1000.0
    
    def update_statistics_display(self, stats: dict):
        """
        Update statistics display.
        
        Args:
            stats: Statistics dictionary.
        """
        self.ui_panel.update_statistics(stats)
    
    def update_interactions_display(self, interactions: list):
        """
        Update current dating interactions display.
        
        Args:
            interactions: List of interaction dictionaries.
        """
        self.ui_panel.update_interactions(interactions)
    
    def update_match_history_display(self, history: list):
        """
        Update match history display.
        
        Args:
            history: List of recent encounter records.
        """
        self.ui_panel.update_match_history(history)
    
    def update_slider_labels(self, speed: float, num_agents: int, agent_speed: float, threshold: float):
        """
        Update slider labels.
        
        Args:
            speed: Current simulation speed.
            num_agents: Current number of agents.
            agent_speed: Current agent movement speed.
            threshold: Current match threshold.
        """
        self.ui_panel.update_labels(speed, num_agents, agent_speed, threshold)
    
    def get_play_pause_button(self):
        """Get play/pause button."""
        return self.ui_panel.play_pause_button
    
    def get_reset_button(self):
        """Get reset button."""
        return self.ui_panel.reset_button
    
    def get_speed_slider(self):
        """Get speed slider."""
        return self.ui_panel.speed_slider
    
    def get_agents_slider(self):
        """Get agents slider."""
        return self.ui_panel.agents_slider
    
    def get_agent_speed_slider(self):
        """Get agent speed slider."""
        return self.ui_panel.agent_speed_slider
    
    def get_threshold_slider(self):
        """Get threshold slider."""
        return self.ui_panel.threshold_slider
    
    def cleanup(self):
        """Cleanup resources."""
        pygame.quit()

