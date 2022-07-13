class Parameters:
    # This should match the file name -- parameters##
    trial_num = 2

    # Domain:
    n_agents = 2
    n_agent_types = 1
    n_pois = 2
    poi_options = [[1, 1, 0]]
    with_agents = True
    size = 30
    time_steps = 100
    n_regions = 8
    sensor_range = 10
    rand_action_rate = 0.05

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
    sigma = 0.1
    learning_rate = 0.03
    n_policies = 50

    # Evo learner
    n_gen = 10000
