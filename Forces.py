import numpy as np
import numpy.typing as npt
from typing import Callable

def continous_force_from_array(forces: npt.ArrayLike, timesteps: npt.ArrayLike, cyclic: bool = True) -> Callable[[float], npt.NDArray]:
    """
    Creates a continous force from an array of forces by linearely interpolation.

    Parameters
    ----------
    forces : array_like
        The forces with shape (6, t) where t is the number of timesteps.
    timesteps : array_like
        The timesteps with shape (t,) corresponding to the forces in the array.
    cyclic : bool, optional
        Whether the force should be cyclic after the last timestep. If false, the function returns a zero-force
        after the last timestep.
        
    Returns
    -------
    Callable[[float], npt.NDArray]
        The continous function that takes time as input and returns the interpolated force.
    """
    forces = np.atleast_2d(forces)
    timesteps = np.atleast_1d(timesteps)
    
    if cyclic:
        def continous_force(time: float) -> npt.NDArray:
            time = time % timesteps[-1]
            return np.array([np.interp(time, timesteps, DOF) for DOF in forces])
    else:
        def continous_force(time: float) -> npt.NDArray:
            if time <= timesteps[-1]:
                return np.array([np.interp(time, timesteps, DOF) for DOF in forces])
            else:
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    return continous_force