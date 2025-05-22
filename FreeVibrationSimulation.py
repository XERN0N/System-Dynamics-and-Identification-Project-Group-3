from OriginalModel import *
from SystemSolvers import *
import matplotlib.pyplot as plt
import pandas as pd

model_secondary = generate_original_model()

with model_secondary.added_forces({8: lambda t: [100, 0, 0, 0, 0, 0]}):
    initial_condition_solver_secondary = Static(model_secondary).solve()

end_time = 15.5
time_increment = 0.0001
solution_secondary = Newmark(model_secondary, initial_condition_solver_secondary, end_time, time_increment).solve().solution

data_secondary = pd.read_csv("Sorted timeseries/Free/1D/Calibrated/weak_axis_data/timeseries_gr3_FirstSoft.csv")
data_primary = pd.read_csv("Sorted timeseries/Free/1D/Calibrated/weak_axis_data/timeseries_gr3_FirstSoft.csv")

fig, axs = plt.subplots(3)

axs[0].plot(data_secondary['Time (us)'], data_secondary['Sensor 4 - X'], label='Experiment')
axs[0].plot(solution_secondary.time, solution_secondary.accelerations[:, Vertex_DOFs.output_secondary_DOFs.value[-1]], label='Model')
axs[0].set_ylabel('Accelerations [$m/s^2$]')
axs[0].legend()
axs[0].set_title('Accelerations of top-most sensor with secondary axis excited')
axs[2].plot(solution_secondary.time, solution_secondary.displacements[:, Vertex_DOFs.output_secondary_DOFs.value[-1]], label='Secondary axis')
axs[2].set_xlabel('Time [$s$]')
axs[2].set_ylabel('Displacements [$m$]')
axs[2].set_title('Model displacements of both axes')


model = generate_original_model()

with model.added_forces({8: lambda t: [0, 130, 0, 0, 0, 0]}):
    initial_condition_solver_primary = Static(model).solve()

solution_primary = Newmark(model, initial_condition_solver_primary, end_time, time_increment).solve().solution

data_secondary = pd.read_csv("Sorted timeseries/Free/1D/Calibrated/strong_axis_data/timeseries_gr3_FirstStiff.csv")

axs[1].plot(data_secondary['Time (us)'], data_secondary['Sensor 4 - X'], label='Experiment')
axs[1].plot(solution_primary.time, solution_primary.accelerations[:, Vertex_DOFs.output_primary_DOFs.value[-1]], label='Model')
axs[1].set_ylabel('Accelerations [$m/s^2$]')
axs[1].legend()
axs[1].set_title('Accelerations of top-most sensor with primary axis excited')
axs[2].plot(solution_primary.time, solution_primary.displacements[:, Vertex_DOFs.output_primary_DOFs.value[-1]], label='Primary axis')
axs[2].legend()
fig.tight_layout()
plt.show()