class Parameters:

    # TODO: MAKE SURE TO CHANGE THIS
    trial_num = 243

    fname_prepend = "G_"    # Domain:
    n_agents = 2
    n_agent_types = 1
    n_pois = 2
    poi_options = [[60, 1, 1, 1], [20, 2, 2, 1.5]]  # time active, number of times active, observation_req, value
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