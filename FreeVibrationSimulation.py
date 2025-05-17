from OriginalModel import generate_original_model
from SystemSolvers import *
import matplotlib.pyplot as plt
import pandas as pd

model = generate_original_model()

with model.added_forces({1: lambda t: [0, 125, 0, 0, 0, 0]}):
    initial_condition_solver = Static(model).solve()

end_time = 15.5
time_increment = 0.00001
solution = Newmark(model, initial_condition_solver, end_time, time_increment).solve().solution

data = pd.read_csv("Sorted timeseries/Free/1D/Calibrated/strong_axis_data/timeseries_gr3_FirstStiff.txt")

plt.plot(data['Time (us)'], data['Sensor 4 - X'], label='Experiment')
plt.plot(solution.time, solution.accelerations[:, 73], label='Model')
plt.xlabel('Time [$s$]')
plt.ylabel('Accelerations [$m/s^2$]')
plt.legend()
plt.show()