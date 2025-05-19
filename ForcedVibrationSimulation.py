from OriginalModel import *
from SystemSolvers import *
import matplotlib.pyplot as plt
import pandas as pd
from Forces import continous_force_from_array
import matplotlib.animation as animation

model = generate_original_model()

initial_condition_solver = Static(model).solve()

# hammer_data = pd.read_csv("Sorted timeseries/Forced/1D/Hammer/Calibrated/X/timeseries_gr3_1D_x_middle2025-03-27-10-54-23.csv.csv", skiprows=range(1, 9750), nrows=25000)

hammer_data = pd.read_csv("Sorted timeseries/Forced/1D/Hammer/Calibrated/X/timeseries_gr3_1D_x_middle_1hit2025-03-27-10-53-34.csv.csv", skiprows=range(1, 5500), nrows=4500)

model.add_forces({5: continous_force_from_array(np.column_stack((hammer_data['Hammer force [N]'], np.zeros((len(hammer_data['Hammer force [N]']), 5)))).T, hammer_data['Time (s)'])})


end_time = 15.5
time_increment = 0.001
scaling_factor = 10000

""" model_forces = []
timesteps = np.arange(0, end_time, time_increment)

for time in timesteps:
    model_forces.append(model.graph.vs[5]['force'](time)[0])

plt.plot(timesteps, model_forces, label='model', linestyle='--')
plt.plot(hammer_data['Time (s)'], hammer_data['Hammer force [N]'], label='Experiment', linestyle='--')

plt.legend()
plt.show()
"""
    
newmark_solver = Newmark(model, initial_condition_solver, end_time, time_increment).solve()

solution = newmark_solver.solution

# plt.plot(accelerometer_data['Time (us)'], accelerometer_data['Sensor 4 - X'], label='accelerometer')
# plt.plot(hammer_data['Time (s)'], hammer_data['Hammer force [N]'], label='Hammer')

displaced_shape_points = newmark_solver.get_displaced_shape_position(scaling_factor)

plot_lines = list()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for displaced_shape_point in displaced_shape_points:
    plot_lines.append(ax.plot(displaced_shape_point[0, :, 0], displaced_shape_point[0, :, 1], displaced_shape_point[0, :, 2], linewidth=2.0))

def update(frame):
    for i, lines in enumerate(plot_lines):
        for line in lines:
            line.set_data_3d(*displaced_shape_points[i][frame, :].T)
    ax.set_title(f"t = {frame*time_increment:.3f} s")
    return plot_lines

ani = animation.FuncAnimation(fig=fig, func=update, frames=int(end_time/time_increment*10), interval=int(time_increment)*10)

ax.axis('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid()
plt.show()

""" plt.plot(solution.time, solution.accelerations[:, Vertex_DOFs.output_secondary_DOFs.value[-1]], label='Model')
plt.xlabel('Time [$s$]')
plt.ylabel('Accelerations [$m/s^2$]')
plt.legend()
plt.show() """