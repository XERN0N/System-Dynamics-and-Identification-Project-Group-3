from OriginalModel import *
from SystemSolvers import *
import matplotlib.pyplot as plt
import pandas as pd
from Forces import continous_force_from_array

model_primary = generate_original_model()

# initial_condition_solver = Static(model).solve()

hammer_data = pd.read_csv("Sorted timeseries/Forced/2D/Hammer/Calibrated/X/timeseries_gr3_2D_middle_2025-03-27-10-22-24.csv.csv")

# model.add_forces({5: continous_force_from_array(np.column_stack((hammer_data['Hammer force [N]'], np.zeros((len(hammer_data['Hammer force [N]']), 5)))).T, hammer_data['Time (s)'])})


# end_time = 15.5
# time_increment = 0.001

""" model_forces = []
timesteps = np.arange(0, end_time, time_increment)

for time in timesteps:
    model_forces.append(model.graph.vs[5]['force'](time)[0])

plt.plot(timesteps, model_forces, label='model', linestyle='--')
plt.plot(hammer_data['Time (s)'], hammer_data['Hammer force [N]'], label='Experiment', linestyle='--')

plt.legend()
plt.show()
"""
    
# solution = Newmark(model, initial_condition_solver, end_time, time_increment).solve().solution

accelerometer_data = pd.read_csv("Sorted timeseries/Forced/2D/Accelerometer/Calibrated/timeseries_serial_output_2_gr3_2D_middle.txt")

plt.plot(accelerometer_data['Time (us)'], accelerometer_data['Sensor 4 - X'], label='accelerometer')
plt.plot(hammer_data['Time (s)'], hammer_data['Hammer force [N]'], label='Hammer')

# plt.plot(accelerometer_data['Time (us)'], accelerometer_data['Sensor 4 - X'], label='Experiment')
# plt.plot(solution.time, solution.accelerations[:, Vertex_DOFs.output_secondary_DOFs.value[-1]], label='Model')
plt.xlabel('Time [$s$]')
plt.ylabel('Accelerations [$m/s^2$]')
plt.legend()
plt.show()