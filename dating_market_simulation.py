"""
Dating Market Simulation - Main Application

An interactive agent-based simulation of a dating market using Pygame.
Integrates Random Forest predictions and Apriori association rules to simulate
dating encounters and match predictions.

Usage:
    python dating_market_simulation.py
    
Author: AIDA Project
Date: November 2025
"""

import sys
import logging
import argparse
import pygame
from controllers.simulation_controller import SimulationController
from views.main_view import MainView
from config.simulation_config import config

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatingMarketSimulation:
    """Main application class for the dating market simulation."""
    
    def __init__(self, num_agents: int = None, speed: float = None):
        """
        Initialize the simulation application.
        
        Args:
            num_agents: Initial number of agents (optional).
            speed: Initial simulation speed (optional).
        """
        logger.info("Initializing Dating Market Simulation")
        
        # Create controller
        self.controller = SimulationController()
        
        # Override initial parameters if provided
        if num_agents is not None:
            self.controller.set_num_agents(num_agents)
        if speed is not None:
            self.controller.set_speed(speed)
        
        # Create view
        self.view = MainView()
        
        # Initialize simulation
        self.controller.initialize_simulation()
        
        # Metrics update counter
        self.metrics_update_counter = 0
        
        # Running state
        self.running = True
        
        logger.info("Dating Market Simulation initialized successfully")
    
    def handle_events(self):
        """Handle all pygame and UI events."""
        events = self.view.process_events()
        
        for event in events:
            # Quit event
            if event.type == pygame.QUIT:
                self.running = False
            
            # UI events from panel
            button_clicked, slider_changed = self.view.ui_panel.handle_event(event)
            
            if button_clicked == 'play_pause':
                self._handle_play_pause()
            elif button_clicked == 'reset':
                self._handle_reset()
            
            if slider_changed == 'speed':
                self._handle_speed_change(self.view.get_speed_slider().value)
            elif slider_changed == 'agents':
                self._handle_agents_change(self.view.get_agents_slider().value)
            elif slider_changed == 'agent_speed':
                self._handle_agent_speed_change(self.view.get_agent_speed_slider().value)
            elif slider_changed == 'threshold':
                self._handle_threshold_change(self.view.get_threshold_slider().value)
    
    def _handle_play_pause(self):
        """Handle play/pause button click."""
        self.controller.toggle_pause()
        
        # Update button text
        button = self.view.get_play_pause_button()
        if self.controller.paused:
            button.text = "Play"
        else:
            button.text = "Pause"
    
    def _handle_reset(self):
        """Handle reset button click."""
        logger.info("Resetting simulation")
        self.controller.reset()
        self.controller.initialize_simulation()
        self.metrics_update_counter = 0
    
    def _handle_speed_change(self, value: float):
        """Handle speed slider change."""
        self.controller.set_speed(value)
        self.view.update_slider_labels(
            value,
            self.controller.num_agents,
            self.controller.agent_speed,
            self.controller.interaction_controller.match_threshold
        )
    
    def _handle_agents_change(self, value: float):
        """Handle agents slider change."""
        self.controller.set_num_agents(int(value))
        self.view.update_slider_labels(
            self.controller.speed,
            int(value),
            self.controller.agent_speed,
            self.controller.interaction_controller.match_threshold
        )
    
    def _handle_agent_speed_change(self, value: float):
        """Handle agent speed slider change."""
        self.controller.set_agent_speed(value)
        self.view.update_slider_labels(
            self.controller.speed,
            self.controller.num_agents,
            value,
            self.controller.interaction_controller.match_threshold
        )
    
    def _handle_threshold_change(self, value: float):
        """Handle threshold slider change."""
        self.controller.set_match_threshold(value)
        self.view.update_slider_labels(
            self.controller.speed,
            self.controller.num_agents,
            self.controller.agent_speed,
            value
        )
    
    def update(self, delta_time: float):
        """
        Update simulation state.
        
        Args:
            delta_time: Time elapsed since last update (seconds).
        """
        # Update UI
        self.view.update_ui(delta_time)
        
        # Update simulation
        self.controller.update(delta_time)
        
        # Update interactions display
        interactions = self.controller.interaction_controller.get_current_interactions()
        self.view.update_interactions_display(interactions)
        
        # Update metrics display periodically
        self.metrics_update_counter += 1
        if self.metrics_update_counter >= config.METRICS_UPDATE_INTERVAL:
            stats = self.controller.get_statistics()
            self.view.update_statistics_display(stats)
            
            # Update match history display
            history = self.controller.metrics_tracker.get_match_history(limit=10)
            self.view.update_match_history_display(history)
            
            self.metrics_update_counter = 0
    
    def render(self):
        """Render the simulation."""
        self.view.render(
            agents=self.controller.agents,
            paused=self.controller.paused
        )
    
    def run(self):
        """Main game loop."""
        logger.info("Starting main game loop")
        
        try:
            while self.running:
                # Get delta time
                delta_time = self.view.tick(config.FPS)
                
                # Handle events
                self.handle_events()
                
                # Update simulation
                self.update(delta_time)
                
                # Render
                self.render()
        
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            raise
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up resources")
        
        # Export metrics if desired
        try:
            self.controller.metrics_tracker.export_to_csv("simulation_results.csv")
            logger.info("Metrics exported to simulation_results.csv")
            
            self.controller.metrics_tracker.export_to_json("simulation_results.json")
            logger.info("Detailed metrics exported to simulation_results.json")
        except Exception as e:
            logger.warning(f"Could not export metrics: {e}")
        
        # Cleanup view
        self.view.cleanup()
        
        logger.info("Cleanup complete")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Dating Market Simulation - Interactive agent-based dating simulation"
    )
    parser.add_argument(
        '--agents',
        type=int,
        default=None,
        help=f'Number of agents ({config.MIN_AGENTS}-{config.MAX_AGENTS}, default: {config.INITIAL_AGENTS})'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=None,
        help=f'Simulation speed multiplier ({config.MIN_SPEED}-{config.MAX_SPEED}, default: {config.DEFAULT_SPEED})'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Print welcome message
    print("=" * 60)
    print("Dating Market Simulation")
    print("=" * 60)
    print("An interactive agent-based simulation using:")
    print("  - Random Forest for match prediction")
    print("  - Apriori association rules for compatibility")
    print("=" * 60)
    print()
    
    try:
        # Create and run simulation
        simulation = DatingMarketSimulation(
            num_agents=args.agents,
            speed=args.speed
        )
        simulation.run()
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        print("\nSimulation stopped by user")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("Check logs for details")
        sys.exit(1)
    
    print("\nSimulation ended. Results exported to simulation_results.csv")
    print("Thank you for using Dating Market Simulation!")


if __name__ == "__main__":
    main()

