"""
Example parameters file for the rover domain.
"""


class Parameters:
    # This should match the file name -- parameters##
    trial_num = 6
    rew_str = 'multi'

    # Domain:
    n_agents = 1
    n_agent_types = 1
    n_poi_types = 2
    rooms = [[50, 50], [50, 50], [80, 20], [20, 80]]
    size = 30
    time_threshold = 15  # How long before information drops out of the state
    time_steps = 50
    sensor_range = 10
    rand_action_rate = 0.05
    n_stat_runs = 1

    # POI:
    value = 1
    obs_radius = 1
    couple = 1
    strong_coupling = False

    # Agent:
    capabilities = False

    # Neural Network:
    hid = 30

    # Evolve nn:
    sigma = 0.5
    learning_rate = 0.1
    n_policies = 300
    # Determines if it evolves half of the species or evolves one third and creates a new set of policies for the other third
    thirds = True

    # Evo learner
    n_gen = 500
    n_top_gen = 500

