class Parameters:

    trial_num = 102

    fname_prepend = "G_"    # Domain:
    n_agents = 3
    n_agent_types = 1
    n_pois = 12
    poi_options = ('exp',)    with_agents = True
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
    learning_rate = 0.05
    n_policies = 50

    # Evo learner
    n_gen = 1500
