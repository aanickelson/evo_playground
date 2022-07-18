class Parameters:
    """
    Big world, few POIs, two types
    """

    # TODO: MAKE SURE TO CHANGE THIS
    trial_num = 25

    # Domain:
    n_agents = 5
    n_agent_types = 1
    n_pois = 15
    poi_options = [[100, 1, 0], [50, 1, 1]]
    with_agents = True
    size = 100
    time_steps = 100
    n_regions = 8
    sensor_range = 10
    rand_action_rate = 0.0

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
    n_policies = 100

    # Evo learner
    n_gen = 5000
