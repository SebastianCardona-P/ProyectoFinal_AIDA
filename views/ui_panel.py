"""
UI Panel for controls and information display.

This module creates the control panel with sliders and buttons using native Pygame.
"""

import pygame
from config.simulation_config import config


class Button:
    """Simple button widget."""
    
    def __init__(self, rect, text, color=(100, 150, 200)):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.color = color
        self.hover_color = tuple(min(c + 30, 255) for c in color)
        self.is_hovered = False
        self.font = pygame.font.Font(None, 24)
    
    def draw(self, surface):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, (50, 50, 50), self.rect, 2, border_radius=5)
        
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                return True
        return False


class Slider:
    """Simple slider widget."""
    
    def __init__(self, rect, min_val, max_val, initial_val, label=""):
        self.rect = pygame.Rect(rect)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.dragging = False
        self.font = pygame.font.Font(None, 20)
    
    def draw(self, surface):
        # Draw label
        if self.label:
            text = self.font.render(self.label, True, (50, 50, 50))
            surface.blit(text, (self.rect.x, self.rect.y - 25))
        
        # Draw track
        track_rect = pygame.Rect(self.rect.x, self.rect.y + 5, self.rect.width, 4)
        pygame.draw.rect(surface, (200, 200, 200), track_rect, border_radius=2)
        
        # Draw handle
        handle_x = self.rect.x + int(
            (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width
        )
        handle_rect = pygame.Rect(handle_x - 8, self.rect.y, 16, 16)
        color = (100, 150, 200) if self.dragging else (80, 120, 180)
        pygame.draw.circle(surface, color, handle_rect.center, 8)
        pygame.draw.circle(surface, (50, 50, 50), handle_rect.center, 8, 2)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            handle_x = self.rect.x + int(
                (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width
            )
            handle_rect = pygame.Rect(handle_x - 8, self.rect.y, 16, 16)
            if handle_rect.collidepoint(event.pos) or self.rect.collidepoint(event.pos):
                self.dragging = True
                self._update_value(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_value(event.pos[0])
            return True
        return False
    
    def _update_value(self, mouse_x):
        rel_x = max(0, min(mouse_x - self.rect.x, self.rect.width))
        self.value = self.min_val + (rel_x / self.rect.width) * (self.max_val - self.min_val)


class UIControlPanel:
    """UI panel with controls and information display."""
    
    def __init__(self, rect: pygame.Rect, surface, config):
        """
        Initialize UI panel.
        
        Args:
            rect: Rectangle for the panel.
            surface: Pygame surface to draw on.
            config: Simulation configuration.
        """
        self.surface = surface
        self.rect = rect
        self.config = config
        
        # Create UI elements
        self._create_controls()
        
        # Statistics text
        self.stats_text = []
        self.font_small = pygame.font.Font(None, 18)
        self.font_title = pygame.font.Font(None, 24)
        self.font_tiny = pygame.font.Font(None, 14)
        
        # Interaction display data
        self.current_interactions = []
        
        # Match history data
        self.match_history = []
        self.history_scroll = 0  # Scroll offset for history
    
    def _create_controls(self):
        """Create control buttons and sliders."""
        y_offset = 60
        x_margin = 20
        width = self.rect.width - 2 * x_margin
        
        # Play/Pause button
        self.play_pause_button = Button(
            (self.rect.x + x_margin, y_offset, width // 2 - 5, 40),
            "Pause"
        )
        
        # Reset button
        self.reset_button = Button(
            (self.rect.x + x_margin + width // 2 + 5, y_offset, width // 2 - 5, 40),
            "Reset",
            color=(180, 100, 100)
        )
        y_offset += 60
        
        # Speed slider
        self.speed_slider = Slider(
            (self.rect.x + x_margin, y_offset, width, 20),
            self.config.MIN_SPEED,
            self.config.MAX_SPEED,
            self.config.DEFAULT_SPEED,
            f"Speed: {self.config.DEFAULT_SPEED:.1f}x"
        )
        y_offset += 50
        
        # Agents slider
        self.agents_slider = Slider(
            (self.rect.x + x_margin, y_offset, width, 20),
            self.config.MIN_AGENTS,
            self.config.MAX_AGENTS,
            self.config.INITIAL_AGENTS,
            f"Agents: {self.config.INITIAL_AGENTS}"
        )
        y_offset += 50
        
        # Agent speed slider
        self.agent_speed_slider = Slider(
            (self.rect.x + x_margin, y_offset, width, 20),
            self.config.MIN_AGENT_SPEED,
            self.config.MAX_AGENT_SPEED,
            self.config.DEFAULT_AGENT_SPEED,
            f"Agent Speed: {self.config.DEFAULT_AGENT_SPEED:.0f} px/s"
        )
        y_offset += 50
        
        # Threshold slider
        self.threshold_slider = Slider(
            (self.rect.x + x_margin, y_offset, width, 20),
            self.config.MIN_MATCH_THRESHOLD,
            self.config.MAX_MATCH_THRESHOLD,
            self.config.DEFAULT_MATCH_THRESHOLD,
            f"Match Threshold: {self.config.DEFAULT_MATCH_THRESHOLD:.2f}"
        )
    
    def draw(self):
        """Draw the UI panel."""
        # Draw panel background
        pygame.draw.rect(self.surface, self.config.COLOR_PANEL_BG, self.rect)
        pygame.draw.rect(self.surface, (200, 200, 200), self.rect, 2)
        
        # Draw title
        title = self.font_title.render("SIMULATION CONTROLS", True, (50, 50, 50))
        self.surface.blit(title, (self.rect.x + 20, self.rect.y + 20))
        
        # Draw buttons and sliders
        self.play_pause_button.draw(self.surface)
        self.reset_button.draw(self.surface)
        self.speed_slider.draw(self.surface)
        self.agents_slider.draw(self.surface)
        self.agent_speed_slider.draw(self.surface)
        self.threshold_slider.draw(self.surface)
        
        # Draw statistics
        stats_y = 320
        stats_title = self.font_title.render("STATISTICS", True, (50, 50, 50))
        self.surface.blit(stats_title, (self.rect.x + 20, stats_y))
        stats_y += 35
        
        for line in self.stats_text:
            text_surf = self.font_small.render(line, True, (70, 70, 70))
            self.surface.blit(text_surf, (self.rect.x + 20, stats_y))
            stats_y += 20
        
        # Draw current interactions
        interaction_y = stats_y + 20
        interaction_title = self.font_title.render("CURRENT DATING", True, (50, 50, 50))
        self.surface.blit(interaction_title, (self.rect.x + 20, interaction_y))
        interaction_y += 35
        
        if self.current_interactions:
            for interaction in self.current_interactions[:2]:  # Show max 2 interactions
                interaction_y = self._draw_interaction(interaction, interaction_y)
        else:
            text = self.font_small.render("No active dates", True, (150, 150, 150))
            self.surface.blit(text, (self.rect.x + 20, interaction_y))
            interaction_y += 25
        
        # Draw match history (if space available)
        if interaction_y < self.rect.height - 50:
            self._draw_match_history(interaction_y + 15)
    
    def _draw_interaction(self, interaction: dict, y_offset: int) -> int:
        """
        Draw a single interaction display.
        
        Args:
            interaction: Dictionary with interaction data.
            y_offset: Y position to start drawing.
            
        Returns:
            New y_offset after drawing.
        """
        x = self.rect.x + 20
        agent1 = interaction['agent1']
        agent2 = interaction['agent2']
        
        # Interaction header
        header = f"Agent {agent1.id} ♥ Agent {agent2.id}"
        text_surf = self.font_small.render(header, True, (255, 100, 100))
        self.surface.blit(text_surf, (x, y_offset))
        y_offset += 22
        
        # Agent attributes (compact)
        attr1 = f"  {agent1.id}: {'M' if agent1.gender else 'F'}, {agent1.age}y, Attr:{agent1.attractiveness:.1f}"
        text_surf = self.font_tiny.render(attr1, True, (80, 80, 80))
        self.surface.blit(text_surf, (x, y_offset))
        y_offset += 16
        
        attr2 = f"  {agent2.id}: {'M' if agent2.gender else 'F'}, {agent2.age}y, Attr:{agent2.attractiveness:.1f}"
        text_surf = self.font_tiny.render(attr2, True, (80, 80, 80))
        self.surface.blit(text_surf, (x, y_offset))
        y_offset += 16
        
        # Match probabilities
        rf_text = f"  RF Prob: {interaction['rf_probability']:.2f}"
        text_surf = self.font_tiny.render(rf_text, True, (50, 100, 200))
        self.surface.blit(text_surf, (x, y_offset))
        y_offset += 16
        
        apriori_text = f"  Apriori Boost: +{interaction['apriori_boost']:.2f}"
        text_surf = self.font_tiny.render(apriori_text, True, (200, 100, 50))
        self.surface.blit(text_surf, (x, y_offset))
        y_offset += 16
        
        # Active rules (show first 2)
        if interaction['active_rules']:
            rules_text = "  Rules: " + ", ".join(interaction['active_rules'][:2])
            if len(rules_text) > 50:
                rules_text = rules_text[:47] + "..."
            text_surf = self.font_tiny.render(rules_text, True, (100, 100, 100))
            self.surface.blit(text_surf, (x, y_offset))
            y_offset += 16
        
        # Final probability
        final_text = f"  FINAL: {interaction['final_probability']:.2f}"
        text_surf = self.font_small.render(final_text, True, (50, 150, 50))
        self.surface.blit(text_surf, (x, y_offset))
        y_offset += 22
        
        # Separator
        pygame.draw.line(self.surface, (200, 200, 200), 
                        (x, y_offset), (x + 400, y_offset), 1)
        y_offset += 8
        
        return y_offset
    
    def _draw_match_history(self, y_offset: int):
        """
        Draw match history section.
        
        Args:
            y_offset: Y position to start drawing.
        """
        x = self.rect.x + 20
        
        # Title
        title = self.font_title.render("MATCH HISTORY", True, (50, 50, 50))
        self.surface.blit(title, (x, y_offset))
        y_offset += 30
        
        if not self.match_history:
            text = self.font_small.render("No matches yet", True, (150, 150, 150))
            self.surface.blit(text, (x, y_offset))
            return
        
        # Show last 5 encounters
        max_to_show = min(5, len(self.match_history))
        available_height = self.rect.height - y_offset - 10
        
        for i in range(max_to_show):
            if y_offset + 40 > self.rect.height - 10:
                break  # No more space
                
            record = self.match_history[i]
            
            # Match indicator
            if record.get('matched', False):
                indicator = "✓"
                color = (50, 180, 50)
            else:
                indicator = "✗"
                color = (180, 50, 50)
            
            # Compact summary line
            summary = f"{indicator} #{record.get('encounter_number', '?')}: {record.get('agent1_id', '?')}+{record.get('agent2_id', '?')} = {record.get('final_probability', 0):.2f}"
            text_surf = self.font_tiny.render(summary, True, color)
            self.surface.blit(text_surf, (x, y_offset))
            y_offset += 15
            
            # Show rules if matched (very compact)
            if record.get('matched') and record.get('apriori_rules'):
                rules = record['apriori_rules']
                if rules:
                    rules_text = f"   Rules: {len(rules)}"
                    text_surf = self.font_tiny.render(rules_text, True, (100, 100, 100))
                    self.surface.blit(text_surf, (x, y_offset))
                    y_offset += 15
            
            y_offset += 5  # Small gap
    
    def handle_event(self, event):
        """
        Handle UI events.
        
        Args:
            event: Pygame event.
            
        Returns:
            Tuple of (button_clicked, slider_changed).
        """
        button_clicked = None
        slider_changed = None
        
        if self.play_pause_button.handle_event(event):
            button_clicked = 'play_pause'
        elif self.reset_button.handle_event(event):
            button_clicked = 'reset'
        
        if self.speed_slider.handle_event(event):
            slider_changed = 'speed'
        elif self.agents_slider.handle_event(event):
            slider_changed = 'agents'
        elif self.agent_speed_slider.handle_event(event):
            slider_changed = 'agent_speed'
        elif self.threshold_slider.handle_event(event):
            slider_changed = 'threshold'
        
        return button_clicked, slider_changed
    
    def update_statistics(self, stats: dict):
        """
        Update statistics display.
        
        Args:
            stats: Dictionary of statistics.
        """
        self.stats_text = [
            f"Total Agents: {stats.get('total_agents', 0)}",
            f"Male/Female: {stats.get('male_agents', 0)}/{stats.get('female_agents', 0)}",
            "",
            f"Encounters: {stats.get('total_encounters', 0)}",
            f"Matches: {stats.get('total_matches', 0)}",
            f"Match Rate: {stats.get('match_rate', 0):.1f}%",
            "",
            f"Dating Now: {stats.get('currently_dating', 0)}",
            f"Matched Agents: {stats.get('matched_agents', 0)}"
        ]
    
    def update_labels(self, speed: float, num_agents: int, agent_speed: float, threshold: float):
        """
        Update slider labels.
        
        Args:
            speed: Current simulation speed.
            num_agents: Current number of agents.
            agent_speed: Current agent movement speed.
            threshold: Current match threshold.
        """
        self.speed_slider.label = f"Simulation Speed: {speed:.1f}x"
        self.agents_slider.label = f"Agents: {int(num_agents)}"
        self.agent_speed_slider.label = f"Agent Speed: {agent_speed:.0f} px/s"
        self.threshold_slider.label = f"Match Threshold: {threshold:.2f}"
    
    def update_interactions(self, interactions: list):
        """
        Update current dating interactions display.
        
        Args:
            interactions: List of interaction dictionaries.
        """
        self.current_interactions = interactions
    
    def update_match_history(self, history: list):
        """
        Update match history display.
        
        Args:
            history: List of recent encounter records (most recent first).
        """
        self.match_history = history
