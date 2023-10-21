"""
Project: Emegence: "Boids" and flocking behaviour
"""

# Libraries used

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import distance_matrix
import time
import io
import os
import imageio # Unused because I have commented the fig.save, to save files on user's computer: uncomment imageio.mimsave line and fig.save lines and create a folder inside the running directory named "plots", the files will automatically be saved there
import datetime

from PIL import Image

current_directory = os.path.dirname(os.path.abspath(__file__))

# Parameters of the simulation

# Boids intrinsic parameters
N = 100                       # Number of boids
speed = 2                     # (Norm of velocity) and cte for all boids
time_step = 1

# Map and miscellaneous parameters
low = -50                       
high = 50 
xlim = [low, high]
turn_value = 0.6
x_turn_lim = 0.75*low
y_turn_lim = 0.75*high
time_simulation = 200
frames = []

# Interaction parameters
R_alignment = 15           # R separation value should be around the same value of boids size in case we want to represent the "real" separation
R_separation = 10            # Radius that englobes the neighbour area for a boid for the separation/alignment/cohesion interaction
R_cohesion = 2              # Leave all R the same value or what values to give each? study difference values impact
weight_alignment = 0.8
weight_separation = 0.25        # weight_interaction is the parameter which rules how intense we want each interaction (one for each of the 3)
weight_cohesion =  1

# For the report
positions_list = []
velocities_average_list = []
positions_average_list = []
sigma_list = []
position_i_boid = np.zeros((time_simulation + 1, 2))

# Definition of functions

def create_boids(N, low, high, speed):
    positions = np.array(np.random.uniform([low,low], [high,high], size = (N,2)))       # Define the x and y limits of our terrain (environment)
    velocities_non_norm = np.array(np.random.uniform(-1, 1, size = (N,2)))
    velocities = speed * normalize_velocity(velocities_non_norm)                                                                                    # Speed is the parameter value we pick for the simulation to be the speed of boids
    distance_matrix_positions = f_distance_matrix_positions(positions)
    return positions, velocities, distance_matrix_positions

def normalize_velocity(velocities_non_norm):
    norm = np.linalg.norm(velocities_non_norm, axis = 1)  
    norm = norm[:, np.newaxis]                                                      # To make it possible to divide the velocities array
    return (velocities_non_norm / norm)

def f_distance_matrix_positions(positions):
    return distance_matrix(positions, positions, p = 2)

def separation_interaction(N, positions, distance_matrix_positions, velocities, R_separation, weight_separation):
    neighbours, positions_neighbours, positions_neighbours_average = [], [], []
    direction_to_average_position, velocity_new = [], []
    for i in range(N):
        neighbours.append(np.where((distance_matrix_positions[i,:] < R_separation) & (distance_matrix_positions[i,:] != 0))) 
        if neighbours[i][0].size > 0:
            positions_neighbours.append(positions[neighbours[i]])
            positions_neighbours_average.append(np.sum(positions_neighbours[i], axis=0) / len(positions_neighbours[i]))
            direction = positions[i] - positions_neighbours_average[i]
            direction_to_average_position.append(direction / np.linalg.norm(direction))
            velocity_new.append(velocities[i] + direction_to_average_position[i] * weight_separation)
        else:
            positions_neighbours.append(0)
            positions_neighbours_average.append(0)
            direction_to_average_position.append(0)
            velocity_new.append(velocities[i])
    velocity_new = normalize_velocity(velocity_new)
    return velocity_new
                       
def alignment_interaction(N, positions, distance_matrix_positions, velocities, R_alignment, weight_alignment):
    neighbours, velocities_neighbours, velocities_neighbours_average = [], [], []
    direction_to_average_velocity, velocity_new = [], []
    
    for i in range(N):
        neighbours.append(np.where((distance_matrix_positions[i,:] < R_alignment) & (distance_matrix_positions[i,:] != 0)))
        if neighbours[i][0].size > 0:
            velocities_neighbours.append(velocities[neighbours[i]])
            velocities_neighbours_average.append(np.sum(velocities_neighbours[i], axis=0) / len(velocities_neighbours[i]))
            direction_to_average_velocity.append(velocities_neighbours_average[i] - velocities[i])
            velocity_new.append(velocities[i] + direction_to_average_velocity[i] * weight_alignment)
        else:
            velocities_neighbours.append(0)
            velocities_neighbours_average.append(0)
            direction_to_average_velocity.append(0)
            velocity_new.append(velocities[i])
    velocity_new = normalize_velocity(velocity_new)
    return velocity_new
    
def cohesion_interaction(N, positions, distance_matrix_positions, velocities, R_cohesion, weight_cohesion):
    neighbours, positions_neighbours, positions_neighbours_average = [], [], []
    direction_to_average_position, velocity_new = [], []
    
    for i in range(N):
        neighbours.append(np.where((distance_matrix_positions[i,:] < R_cohesion) & (distance_matrix_positions[i,:] != 0)))
        if neighbours[i][0].size > 0:
            positions_neighbours.append(positions[neighbours[i]])
            positions_neighbours_average.append(np.sum(positions_neighbours[i], axis=0) / len(positions_neighbours[i][0]))
            direction = positions_neighbours_average[i] - positions[i]
            direction_to_average_position.append(direction / np.linalg.norm(direction))
            velocity_new.append(velocities[i] + direction_to_average_position[i] * weight_cohesion)
        else:
            positions_neighbours.append(0)
            positions_neighbours_average.append(0)
            direction_to_average_position.append(0)
            velocity_new.append(velocities[i])
    velocity_new = normalize_velocity(velocity_new)
    return velocity_new

def limit_position_inside_map_toroidal(positions, low, high):
    width = high - low
    positions = np.where(positions > high, positions - width, positions)
    positions = np.where(positions < low, positions + width, positions)
    return positions

def limit_position_inside_map_reflective(positions, velocities, x_turn_lim, y_turn_lim, turn_value):
    velocities[:,0] = np.where(positions[:,0] > x_turn_lim, velocities[:,0] - turn_value, velocities[:,0]) 
    velocities[:,0] = np.where(positions[:,0] < -x_turn_lim, velocities[:,0] + turn_value, velocities[:,0])
    velocities[:,1] = np.where(positions[:,1] > y_turn_lim, velocities[:,1] - turn_value, velocities[:,1])
    velocities[:,1] = np.where(positions[:,1] < -y_turn_lim, velocities[:,1] + turn_value, velocities[:,1])
    return velocities
    
def plot_function(positions, velocities, low, high, positions_average, velocities_average, i):
    
    fig, ax = plt.subplots()
    ax.quiver(positions[:, 0], positions[:, 1], velocities[:, 0], velocities[:, 1])
    ax.quiver(positions_average[0], positions_average[1], velocities_average[0], velocities_average[1], color='r')    
    circle = Circle(positions_average, radius = R_alignment, fill = False, ec='r')
    ax.add_patch(circle)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal")
    props = dict(facecolor='white', alpha=0.5)
    textstr = '\n'.join([f"N = {N}", f"Speed = {speed}", f'Radius_a: {R_alignment}', f'Radius_c: {R_cohesion}', f"Radius_s: {R_separation}", f'Weight_a: {weight_alignment}', f"Weight_c: {weight_cohesion}", f"Weight_s: {weight_separation}"])
    ax.text(1.06, 0.8, textstr, transform=ax.transAxes, fontsize=10,
    verticalalignment='top', bbox=props)
    #plt.show()
    
    # To store the plots for the gif
    buf = io.BytesIO()
    fig.savefig(buf, format = "png", dpi = 140)
    buf.seek(0)
    img = (Image.open(buf))
    
    ax.clear()
    plt.close()
    return img

# End of the definition of functions

# Call the function that creates the boids to initialise them
positions, velocities, distance_matrix_positions = create_boids(N, low, high, speed)
positions_average = np.sum(positions, axis = 0) / len(positions) # Is = positions CM

#) This part is only for my report paper
positions_list.append(positions)
positions_average_list.append(positions_average)
velocities_average = np.sum(velocities, axis = 0) / len(velocities)
velocities_average_list.append(velocities_average)
sigma_list.append(np.std(np.array(positions), axis = 0))
# Until here is for the report

# Check if folder inside current directory exists, if not create it
#current_directory = os.path.dirname(os.path.abspath(__file__))
folder_name = "plots_new"
folder_path  = os.path.join(current_directory, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"{folder_name} folder created.")
else:
    print(f"{folder_name} folder already exists.")


# Iteration over time where the rest of the functions are called at each iteration

for i in range(time_simulation):  
    velocity_separation = separation_interaction(N, positions, distance_matrix_positions, velocities, R_separation, weight_separation)
    velocity_alignment = alignment_interaction(N, positions, distance_matrix_positions, velocities, R_alignment, weight_alignment)
    velocity_cohesion = cohesion_interaction(N, positions, distance_matrix_positions, velocities, R_cohesion, weight_cohesion)
    
    velocities += velocity_separation + velocity_alignment + velocity_cohesion                                                                                      
    velocity_direction = normalize_velocity(velocities)
    velocities = speed * velocity_direction
    positions += velocities * time_step
    
    # Here you comment/uncomment one of the two lines below to choose the map behaviour
    positions = limit_position_inside_map_toroidal(positions, low, high)
    #velocities = limit_position_inside_map_reflective(positions, velocities, x_turn_lim, y_turn_lim, turn_value)
    
    distance_matrix_positions = f_distance_matrix_positions(positions)
    
    positions_list.append(positions) 
    velocities_average = np.sum(velocities, axis = 0) / len(velocities)
    velocities_average_list.append(velocities_average)
    positions_average_list.append(positions_average)
    positions_average = np.sum(positions, axis = 0) / len(positions)                                                                                                             
    sigma_list.append(np.std(np.array(positions), axis = 0))
    
    
    frames.append(plot_function(positions, velocities, low, high, positions_average, velocities_average, i))
    time.sleep(0.0001)
    
now = datetime.datetime.now()
imageio.mimsave(current_directory + f"\\{folder_name}" + f"/boids_test{now:%Y-%m-%d_%H-%M-%S}.gif", frames, fps = 7, duration = 0.1)
       
plt.close() 


# Below here is for the report 

# sigma_array = np.array(sigma_list)
# positions_average_array = np.array(positions_average_list)
# velocities_average_array = np.array(velocities_average_list)
# change_average_velocities = np.array([abs(velocities_average_array[i+1] - velocities_average_array[i]) for i in range(len(velocities_average_array)-1)])
# x_units = np.arange(0, time_simulation + 1, 1)


# # Plots for the report

# #Sigma (std) 
# fig, ax = plt.subplots()
# # ax.plot(x_units, sigma_array[:,0])
# # ax.plot(x_units, sigma_array[:,1])
# ax.plot(x_units, np.sqrt(sigma_array[:,0]**2 + sigma_array[:,1]**2))
# ax.set(xscale='log')
# ax.set_xlabel("log(t)")
# ax.set_ylabel("standard deviation")
# # plt.legend()
# plt.show()
# fig.savefig(current_directory + "\\plots" f"/trajectories_respect_CM{now:%Y-%m-%d_%H-%M-%S}.png", dpi = 180)

# fig, ax = plt.subplots()
# ax.plot(positions_average_array[:,0], positions_average_array[:,1])
# ax.plot(positions_average_array[0,0], positions_average_array[0,1], "o", c = "r", markersize = 5, label = "Start")
# ax.plot(positions_average_array[-1,0], positions_average_array[-1,1], "o", c = "g", markersize = 5, label = "End")
# ax.set_xlabel("t")
# plt.legend()
# plt.show()
# fig.savefig(current_directory + "\\plots" f"/trajectories_average_position_plot{now:%Y-%m-%d_%H-%M-%S}.png", dpi = 180)

# fig, ax = plt.subplots()
# ax.plot(x_units, velocities_average_array[:,0] + velocities_average_array[:,1], "+-", c = "r", markersize = 0.5)
# ax.set_xlabel("t")
# ax.set_ylabel("v_total")
# plt.legend()
# plt.show()
# fig.savefig(current_directory + "\\plots" f"/plot_vx_vy{now:%Y-%m-%d_%H-%M-%S}.png", dpi = 180)

# fig, ax = plt.subplots()
# ax.plot(x_units[:-1], change_average_velocities[:,0], "+-", c = "g", markersize = 1, label = "change vx")
# ax.plot(x_units[:-1], change_average_velocities[:,1], "+-", c = "r", markersize = 0.5, label = "change vy")
# ax.set_xlabel("t")
# ax.set_ylabel("velocity")
# plt.legend()
# plt.show()
# fig.savefig(current_directory + "\\plots" + f"/plot_change_vx_vy{now:%Y-%m-%d_%H-%M-%S}.png", dpi = 180)
