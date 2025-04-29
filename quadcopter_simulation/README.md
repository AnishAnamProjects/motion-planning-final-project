# Multi-Agent Drone Exploration Simulations

This repository provides two simulation frameworks for multi-agent drone exploration:

- **A\***: grid-based path planning with role-based exploration and task execution.
- **RRT\***: continuous-space Rapidly-exploring Random Tree-Star planning integrated into the same multi-agent setup.

## To Run

The simulator only uses matplotlib and scipy. You can simply run the following:
```
make
python runsim.py
```

## Comparing A* vs RRT*

Run both planners back-to-back and compare their performance:

```bash
python run_rrt_simulation.py \
  --compare       \  # triggers A* then RRT* simulations
  --surveyors 3   \
  --workers 2     \
  --space 10.0
```

This will:
1. Run the A* simulation and print its metrics.
2. Run the RRT* simulation and print its metrics.
3. Generate side-by-side comparison plots (`astar_vs_rrtstar.png`).

## Customization

- **Surveyors vs Workers:** Adjust roles with `--surveyors` and `--workers`.
- **Environment Size:** Change `--space` to vary the world bounds.
- **Debug Visualization:** Toggle per-frame logs with `--no-debug`.

Happy exploring! 🚁

