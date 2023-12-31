# Boids-algorithm

## Boids simulation
In 1986 Craig Reynolds published the paper "Flocks, Herds, and Schools: A Distributed Behavioral Model", in which he explored how simple rules applied to individuals could lead to the
emergence of collective motion at large scale.

The study of the boids (shortened version of "bird-oid") model has a multitude of applications ranging from the study of prey-predator relationships in ecosystems, simulations
of a crowd of people at a stampede in case of a terrorist attack, programming drone swarms,
animation for the movement of flocks and herds in movies and video games...

The aim of this project is to build a boids simulation from scratch in python. The simulation is used to study the emergence of collective behavior and patterns, exploring
how different parameters affect the dynamics of the system.


![Example of the simulation for N = 10 000 boids](./simulation_N_1000.gif)

## Improvements
This was my first ever programming project, part of my numerical physics course. Having just 4 months of experience in python there is plenty of room for improvements. 

- Implement it in fixed point instead to considerably increase performance and cut computational costs the program.
- Extend to 3D simulation following the same theoretical principles and approach followed in the 2D simulation.
- Use sliders to change the parameters in real time (with ipywidgets).
- Include a boid predator that follows the boids and makes boids disappear as it touches them. This predator would have a different radius than those set for the boids and
would move at a higher speed. It would be interesting to see if the flocking behaviour prolong their survival time and what parameters do it best.
- Give the simulation a more realistic touch: obstacles that boids have to avoid, food that they go after, allow collisions between boids and change the constant speed for speed that could range within an interval.
