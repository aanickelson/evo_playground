class Parameters:

    trial_num = 14

    stat_runs = 50
    fname_prepend = "G_"    # Domain:
    n_agents = 2
    n_agent_types = 1
    n_pois = 3
    poi_options = ('on', 'on', 'on')
    offset = True
    dist_to_poi = 3
    with_agents = True
    size = 10
    time_steps = 50
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
    learning_rate = 0.05
    n_policies = 50

    # Evo learner
    n_gen = 1500
