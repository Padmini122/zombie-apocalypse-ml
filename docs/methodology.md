# Methodology: Zombie Apocalypse Supply Chain Optimizer

## Problem Statement
Predict which evacuation routes will fail during a simulated zombie outbreak to help survivors find the safest path.

## Approach

### 1. Simulation
- Created a network of 20 cities connected by highways
- Each city has attributes: population, elevation, walls, supplies, weapons
- Simulated zombie spread over 30 days using a probability-based model
- Infection spreads faster: lower elevation, higher population, shorter distance

### 2. Feature Engineering
- Created