class Parameters:
    """
    Big world, few POIs, three types (one with 2 visits)
    P98 uses the entropy of the reward vector to update the evolutionary method
    P99 uses G
    """

    # TODO: MAKE SURE TO CHANGE THIS
    trial_num = 97
    fname_prepend = "G_"

    # Domain:
    n_agents = 3
    n_agent_types = 1
    n_pois = 5
    poi_options = [[1, 60, 1, 1]]       # time active, number of times active, observation_req, value
    with_agents = True
    size = 10
    time_steps = 60
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
    learning_rate = 0.05
    n_policies = 50

    # Evo learner
    n_gen = 1000

