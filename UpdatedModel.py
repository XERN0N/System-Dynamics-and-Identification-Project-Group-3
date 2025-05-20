from SystemModels import Beam_Lattice
from Model_updating import newton_update

def generate_updated_model() -> Beam_Lattice:
    """
    Generates the updated model.

    Returns
    -------
    Beam_Lattice
        The updated model.
    """
    parameters = {'cross_sectional_area': None, 'point_mass': None}
    target_frequencies = [26.704, 37.07, 225.315]
    target_frequency_indices = [0, 1, 2]
    updated_model, _, _ = newton_update(parameters, target_frequencies, target_frequency_indices)
    return updated_model