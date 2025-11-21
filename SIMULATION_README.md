# Dating Market Simulation - User Guide

## Overview

An interactive agent-based simulation of a dating market using Pygame. This simulation integrates machine learning (Random Forest) and data mining (Apriori association rules) to create a realistic dating market environment.

## Features

- **Agent-Based Modeling**: Simulates individual agents with realistic attributes from the Speed Dating dataset
- **ML Integration**: 
  - Random Forest model for match prediction
  - Apriori association rules for compatibility boost
- **Interactive Visualization**: Real-time 2D simulation with moving agents
- **Dynamic Controls**: 
  - Play/Pause simulation
  - Reset simulation
  - Adjust simulation speed (0.5x - 3.0x)
  - Change number of agents (10 - 200)
  - Modify match threshold (0.3 - 0.9)
- **Real-Time Statistics**:
  - Total agents (male/female breakdown)
  - Number of encounters
  - Number of matches
  - Match rate percentage
  - Currently dating agents
  - Matched agents

## Installation

### Prerequisites

- Python 3.11.9 or compatible version
- Virtual environment (recommended)

### Setup

1. **Activate the virtual environment**:
   ```powershell
   .\.venv\Scripts\Activate
   ```

2. **Install dependencies** (if not already installed):
   ```powershell
   pip install -r requirements.txt
   ```

   Key dependencies:
   - `pygame` - Visualization and UI
   - `scikit-learn` - Machine learning models
   - `pandas` - Data processing
   - `numpy` - Numerical operations
   - `joblib` - Model loading

## Running the Simulation

### Basic Usage

```powershell
python dating_market_simulation.py
```

### With Parameters

```powershell
# Start with 50 agents
python dating_market_simulation.py --agents 50

# Start with 2x speed
python dating_market_simulation.py --speed 2.0

# Combine parameters
python dating_market_simulation.py --agents 75 --speed 1.5
```

### Command-Line Arguments

- `--agents N`: Initial number of agents (10-200, default: 50)
- `--speed X`: Simulation speed multiplier (0.5-3.0, default: 1.0)

## How to Use the Simulation

### Understanding the Visualization

**Agent Colors**:
- **Blue circles** = Male agents (idle)
- **Pink circles** = Female agents (idle)
- **Yellow circles** = Agents currently dating
- **Green circles** = Agents who have matched
- **Yellow lines** = Active dating encounters

**Agent Sizes**: Circle size represents attractiveness level

### Controls

**Play/Pause Button**: Toggle simulation on/off
**Reset Button**: Restart simulation with new agents

**Sliders**:
- **Speed**: Control simulation speed (0.5x to 3.0x)
- **Agents**: Change number of agents (requires reset to take effect)
- **Match Threshold**: Adjust minimum probability for a successful match (0.3 to 0.9)

### Simulation Mechanics

1. **Movement**: Agents move randomly within the canvas area
2. **Encounters**: When agents get close enough, a dating encounter is initiated
3. **Match Prediction**: 
   - Random Forest model predicts match probability based on agent attributes
   - Association rules provide compatibility boost
   - If combined score exceeds threshold → MATCH!
4. **States**:
   - **IDLE**: Searching for potential dates
   - **DATING**: Currently in an encounter (3-5 seconds)
   - **MATCHED**: Successfully matched and paired

## Technical Architecture

### Project Structure

```
ProyectoFinal_AIDA/
├── config/
│   └── simulation_config.py        # Configuration parameters
├── models/
│   ├── agent.py                    # Agent class and factory
│   ├── predictor.py                # Random Forest predictor
│   └── rules_engine.py             # Apriori rules evaluator
├── controllers/
│   ├── interaction_controller.py   # Dating encounter management
│   └── simulation_controller.py    # Main simulation orchestrator
├── views/
│   ├── agent_renderer.py           # Agent visualization
│   ├── ui_panel.py                 # Control panel UI
│   └── main_view.py                # Main window coordination
├── utils/
│   ├── data_loader.py              # ML model and data loaders
│   ├── collision_detector.py       # Spatial collision detection
│   └── metrics_tracker.py          # Statistics tracking
├── dating_market_simulation.py     # Main application
└── Speed_Dating_Data_Cleaned.csv   # Dataset
```

### ML Models Used

**Random Forest Model**:
- Location: `decision_tree_results/models/random_forest_model.pkl`
- Features: 195 features from Speed Dating dataset
- Output: Match probability (0.0 - 1.0)

**Apriori Association Rules**:
- Location: `apriori_results/data/association_rules.csv`
- Provides compatibility boost based on heuristic rules
- Boost range: 0.0 - 0.4

### Key Components

**Agent System**:
- Realistic demographics (age, race, field of study)
- Personality attributes (attractiveness, intelligence, fun, etc.)
- State machine (IDLE → DATING → MATCHED)
- Movement with boundary constraints

**Collision Detection**:
- Spatial hash grid for O(n) performance
- Interaction radius: 30 pixels

**Match Prediction**:
- Combined score = RF probability + rules boost
- Prediction caching for performance
- Configurable match threshold

## Output

When the simulation ends, results are exported to:
- **simulation_results.csv**: Detailed metrics over time

## Troubleshooting

### Pygame window doesn't open
- Ensure pygame is installed: `pip install pygame`
- Check for display/graphics driver issues

### Models not loading
- Verify Random Forest model exists: `decision_tree_results/models/random_forest_model.pkl`
- Verify association rules exist: `apriori_results/data/association_rules.csv`

### Performance issues
- Reduce number of agents (use `--agents 30`)
- Lower simulation speed
- Close other applications

### Import errors
- Activate virtual environment: `.\.venv\Scripts\Activate`
- Reinstall dependencies: `pip install -r requirements.txt`

## Performance Optimization

The simulation uses several optimizations:
- **Spatial hashing** for collision detection (O(n) instead of O(n²))
- **Prediction caching** to avoid redundant ML calls
- **Limited rules loading** (1000 top rules only)
- **Encounter cooldown** to prevent duplicate interactions

## Development Notes

### Design Principles

- **SOLID**: Single responsibility, open/closed, dependency inversion
- **DRY**: Reusable components (data loaders, renderers)
- **KISS**: Simple, straightforward implementations
- **MVC Pattern**: Separation of models, views, controllers

### Extending the Simulation

**Adding new agent attributes**:
1. Update `AgentFactory` in `models/agent.py`
2. Modify feature vector in `predictor.py`

**Adding new rules**:
1. Update `AssociationRulesEngine.evaluate_rules()` in `models/rules_engine.py`

**Customizing visualization**:
1. Modify `AgentRenderer` in `views/agent_renderer.py`
2. Update colors in `config/simulation_config.py`

## Credits

**Dataset**: Speed Dating Experiment Dataset (8,378 records)
**ML Models**: Random Forest (scikit-learn), Apriori (mlxtend)
**Framework**: Pygame 2.6.1
**Author**: AIDA Project Team
**Date**: November 2024

## License

Educational project for AIDA course.

---

**Enjoy exploring the dating market simulation! 💘**
