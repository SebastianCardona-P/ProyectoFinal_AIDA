"""
Metrics tracking for the simulation.

This module tracks and analyzes simulation metrics over time.
"""

import logging
from typing import Dict, List, Any
from collections import deque
import csv
import json

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and analyze simulation metrics."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics tracker.
        
        Args:
            max_history: Maximum number of historical data points to keep.
        """
        self.max_history = max_history
        
        # Time-series data
        self.encounters_over_time = deque(maxlen=max_history)
        self.matches_over_time = deque(maxlen=max_history)
        self.match_rate_over_time = deque(maxlen=max_history)
        
        # Counters
        self.total_encounters = 0
        self.total_matches = 0
        
        # Detailed records
        self.encounter_records: List[Dict[str, Any]] = []
        
        # Aggregated metrics
        self.matches_by_field: Dict[int, int] = {}
        self.matches_by_race: Dict[int, int] = {}
        self.matches_by_age_gap: Dict[str, int] = {}
        
    def record_encounter(self, 
                        agent1_id: int,
                        agent2_id: int,
                        matched: bool,
                        probability: float,
                        compatibility_boost: float = 0.0,
                        rf_probability: float = None,
                        active_rules: List[str] = None,
                        agent1_attrs: Dict = None,
                        agent2_attrs: Dict = None):
        """
        Record a dating encounter.
        
        Args:
            agent1_id: First agent ID.
            agent2_id: Second agent ID.
            matched: Whether the encounter resulted in a match.
            probability: Final combined match probability.
            compatibility_boost: Boost from association rules.
            rf_probability: Random Forest prediction probability.
            active_rules: List of Apriori rules that fired.
            agent1_attrs: Dictionary of agent1 attributes.
            agent2_attrs: Dictionary of agent2 attributes.
        """
        import time
        
        self.total_encounters += 1
        
        if matched:
            self.total_matches += 1
        
        # Record details
        record = {
            'timestamp': time.time(),
            'agent1_id': agent1_id,
            'agent2_id': agent2_id,
            'matched': matched,
            'rf_probability': rf_probability if rf_probability is not None else probability - compatibility_boost,
            'apriori_boost': compatibility_boost,
            'apriori_rules': active_rules if active_rules else [],
            'final_probability': probability,
            'encounter_number': self.total_encounters
        }
        
        # Add agent attributes if provided
        if agent1_attrs:
            for key, value in agent1_attrs.items():
                record[f'agent1_{key}'] = value
        if agent2_attrs:
            for key, value in agent2_attrs.items():
                record[f'agent2_{key}'] = value
        
        self.encounter_records.append(record)
        
        # Update time series
        self.encounters_over_time.append(self.total_encounters)
        self.matches_over_time.append(self.total_matches)
        
        # Calculate match rate
        match_rate = (self.total_matches / self.total_encounters * 100) if self.total_encounters > 0 else 0
        self.match_rate_over_time.append(match_rate)
    
    def record_match_details(self, 
                            field1: int,
                            field2: int,
                            race1: int,
                            race2: int,
                            age_gap: int):
        """
        Record detailed match information.
        
        Args:
            field1: First agent's field.
            field2: Second agent's field.
            race1: First agent's race.
            race2: Second agent's race.
            age_gap: Age difference.
        """
        # Track by field
        self.matches_by_field[field1] = self.matches_by_field.get(field1, 0) + 1
        self.matches_by_field[field2] = self.matches_by_field.get(field2, 0) + 1
        
        # Track by race
        self.matches_by_race[race1] = self.matches_by_race.get(race1, 0) + 1
        self.matches_by_race[race2] = self.matches_by_race.get(race2, 0) + 1
        
        # Track by age gap category
        if age_gap <= 2:
            category = "0-2 years"
        elif age_gap <= 5:
            category = "3-5 years"
        elif age_gap <= 10:
            category = "6-10 years"
        else:
            category = "10+ years"
        
        self.matches_by_age_gap[category] = self.matches_by_age_gap.get(category, 0) + 1
    
    def get_match_rate(self, time_window: int = 0) -> float:
        """
        Get current match rate.
        
        Args:
            time_window: Number of recent encounters to consider (0 for all).
            
        Returns:
            Match rate as percentage.
        """
        if self.total_encounters == 0:
            return 0.0
        
        if time_window > 0 and len(self.encounter_records) > 0:
            # Calculate for recent window
            recent = self.encounter_records[-time_window:]
            matches = sum(1 for r in recent if r['matched'])
            return (matches / len(recent) * 100) if recent else 0.0
        
        # Overall rate
        return (self.total_matches / self.total_encounters * 100)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics.
        
        Returns:
            Dictionary of statistics.
        """
        stats = {
            'total_encounters': self.total_encounters,
            'total_matches': self.total_matches,
            'match_rate': self.get_match_rate(),
            'matches_by_field': dict(self.matches_by_field),
            'matches_by_race': dict(self.matches_by_race),
            'matches_by_age_gap': dict(self.matches_by_age_gap)
        }
        
        # Calculate average probability
        if self.encounter_records:
            avg_prob = sum(r.get('final_probability', 0) for r in self.encounter_records) / len(self.encounter_records)
            stats['average_probability'] = avg_prob
        
        return stats
    
    def get_time_series_data(self, max_points: int = 100) -> Dict[str, List]:
        """
        Get time series data for plotting.
        
        Args:
            max_points: Maximum number of points to return.
            
        Returns:
            Dictionary with time series data.
        """
        # Sample data if too many points
        encounters = list(self.encounters_over_time)
        matches = list(self.matches_over_time)
        rates = list(self.match_rate_over_time)
        
        if len(encounters) > max_points:
            step = len(encounters) // max_points
            encounters = encounters[::step]
            matches = matches[::step]
            rates = rates[::step]
        
        return {
            'encounters': encounters,
            'matches': matches,
            'match_rates': rates
        }
    
    def get_match_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent match history.
        
        Args:
            limit: Maximum number of recent encounters to return.
            
        Returns:
            List of encounter records (most recent first).
        """
        return list(reversed(self.encounter_records[-limit:]))
    
    def export_to_csv(self, filename: str):
        """
        Export encounter records to CSV.
        
        Args:
            filename: Output filename.
        """
        if not self.encounter_records:
            logger.warning("No encounter records to export")
            return
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                # Get all possible fieldnames from all records
                all_keys = set()
                for record in self.encounter_records:
                    all_keys.update(record.keys())
                fieldnames = sorted(all_keys)
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for record in self.encounter_records:
                    # Convert list fields to strings
                    row = {}
                    for key, value in record.items():
                        if isinstance(value, list):
                            row[key] = '|'.join(str(v) for v in value)
                        else:
                            row[key] = value
                    writer.writerow(row)
            
            logger.info(f"Exported {len(self.encounter_records)} records to {filename}")
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
    
    def export_to_json(self, filename: str):
        """
        Export detailed encounter records to JSON.
        
        Args:
            filename: Output filename.
        """
        if not self.encounter_records:
            logger.warning("No encounter records to export")
            return
        
        try:
            data = {
                'summary': {
                    'total_encounters': self.total_encounters,
                    'total_matches': self.total_matches,
                    'match_rate': self.get_match_rate()
                },
                'encounters': self.encounter_records
            }
            
            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=2)
            
            logger.info(f"Exported {len(self.encounter_records)} records to {filename}")
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
    
    def reset(self):
        """Reset all metrics."""
        self.encounters_over_time.clear()
        self.matches_over_time.clear()
        self.match_rate_over_time.clear()
        
        self.total_encounters = 0
        self.total_matches = 0
        
        self.encounter_records.clear()
        
        self.matches_by_field.clear()
        self.matches_by_race.clear()
        self.matches_by_age_gap.clear()
        
        logger.info("Metrics reset")
