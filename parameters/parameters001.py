class Parameters:

    trial_num = 1
    stat_runs = 1
    fname_prepend = "G_"    # Domain:
    n_agents = 2
    n_agent_types = 1
    n_pois = 4
    poi_options = ('square',)
    with_agents = True
    size = 20
    time_steps = 60
    n_regions = 8
    sensor_range = 10
    rand_action_rate = 0.05

    # POI:
    value = 1
    obs_radius = 1
    couple = 2
    strong_coupling = False

    # Agent:
    capabilities = False

    # Neural Network:
    hid = 30

    # Evolve nn:
    sigma = 0.1
    learning_rate = 0.05
    n_policies = 30

    # Evo learner
    n_gen = 100
