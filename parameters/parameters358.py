class Parameters:

    # TODO: MAKE SURE TO CHANGE THIS
    trial_num = 358

    fname_prepend = "G_"    # Domain:
    n_agents = 3
    n_agent_types = 1
    n_pois = 3
    poi_options = [[20, 3, 1, 1]]  # time active, number of times active, observation_req, value
    with_agents = True
    size = 20
    time_steps = 60
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
    n_gen = 3000
