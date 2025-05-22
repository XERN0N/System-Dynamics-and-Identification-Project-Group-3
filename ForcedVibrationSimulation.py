from OriginalModel import *
from SystemSolvers import *
import matplotlib.pyplot as plt
import pandas as pd
from Forces import continous_force_from_array
import matplotlib.animation as animation
from FFT_Plotter import Signal_integration

model = generate_original_model()

initial_condition_solver = Static(model).solve()

hammer_data = pd.read_csv("Sorted timeseries/Test gruppe 1/Hammer kalibreret/timeseries_hammer_gr1_y_dir_160_4hit.csv.csv", skiprows=range(1, 14130))
hammer_data['Time (s)'] = hammer_data['Time (s)'] - hammer_data['Time (s)'][0]
# hammer_data = pd.read_csv("Sorted timeseries/Forced/1D/Hammer/Calibrated/X/timeseries_gr3_1D_x_middle_1hit2025-03-27-10-53-34.csv.csv", skiprows=range(1, 5500), nrows=10000)

acceleration_data = pd.read_csv("Sorted timeseries/Test gruppe 1/Accelerometer kalibreret/timeseries_gr1_y_dir_160_4hit.csv")

hammer_force = continous_force_from_array(np.column_stack((np.zeros((len(hammer_data['Hammer force [N]']), 1)), hammer_data['Hammer force [N]'], np.zeros((len(hammer_data['Hammer force [N]']), 4)))).T, hammer_data['Time (s)'], 
                                          cyclic=False)

model.add_forces({5: hammer_force})


end_time = 16.5
time_increment = 0.001
# scaling_factor = 1

""" model_forces = []
timesteps = np.arange(hammer_data['Time (s)'][0], hammer_data['Time (s)'][0] + end_time, time_increment)

for time in timesteps:
    model_forces.append(model.graph.vs[5]['force'](time)[0])

plt.plot(timesteps, model_forces, label='model', linestyle='--', marker='o')
plt.plot(hammer_data['Time (s)'], hammer_data['Hammer force [N]'], label='Experiment', linestyle='--', marker='o')

plt.legend()
plt.show() """

""" plt.plot(hammer_data['Time (s)'], hammer_data['Hammer force [N]'], label='Experiment', linestyle='--')
plt.plot(acceleration_data['Time (us)'], acceleration_data['Sensor 4 - X'], label='Experiment', linewidth=0.1)
plt.show() """

    
newmark_solver = Newmark(model, initial_condition_solver, end_time, time_increment).solve()

solution = newmark_solver.solution

# plt.plot(accelerometer_data['Time (us)'], accelerometer_data['Sensor 4 - X'], label='accelerometer')
# plt.plot(hammer_data['Time (s)'], hammer_data['Hammer force [N]'], label='Hammer')

""" displaced_shape_points = newmark_solver.get_displaced_shape_position(scaling_factor)

plot_lines = list()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for displaced_shape_point in displaced_shape_points:
    plot_lines.append(ax.plot(displaced_shape_point[0, :, 0], displaced_shape_point[0, :, 1], displaced_shape_point[0, :, 2], linewidth=2.0))

animation_time_increment = 0.1

def update(frame):
    for i, lines in enumerate(plot_lines):
        for line in lines:
            line.set_data_3d(*displaced_shape_points[i][frame, :].T)
    ax.set_title(f"t = {frame*animation_time_increment:.3f} s")
    return plot_lines

ani = animation.FuncAnimation(fig=fig, func=update, frames=int(end_time/animation_time_increment), interval=animation_time_increment*1000)

ax.axis('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid()
plt.show() """

""" delta_t = 1.079e-3
Velocity_signal = Signal_integration("Velocity", acceleration_data['Sensor 4 - X'], delta_t)
Velocity_signal -= np.mean(Velocity_signal)
Displacement_signal = Signal_integration("Displacement", Velocity_signal, delta_t)
Displacement_signal -= np.mean(Displacement_signal) """

fig, axs = plt.subplots(2)
axs[0].plot(solution.time, solution.accelerations[:, Vertex_DOFs.output_primary_DOFs.value[-1]], label='Model', linewidth=1.0)
axs[1].plot(solution.time, solution.displacements[:, Vertex_DOFs.output_primary_DOFs.value[-1]], label='Model', linewidth=1.0)
axs[0].plot(acceleration_data['Time (us)'], acceleration_data['Sensor 4 - X'], label='Experiment', linewidth=1.0)
axs[1].set_xlabel('Time [$s$]')
axs[0].set_ylabel('Accelerations [$m/s^2$]')
axs[1].set_ylabel('Displacements [$m$]')
axs[0].legend()
axs[1].legend()
plt.show()