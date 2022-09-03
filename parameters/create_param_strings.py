
def string_to_save(i, n_agents, n_pois, poi_options):
    s = ""

    s += "class Parameters:\n\n"

    s += f"    trial_num = {i}\n\n"
    s += f"    stat_runs = 10\n"
    s += f'    fname_prepend = "G_"'
    s += f"    # Domain:\n"
    s += f"    n_agents = {n_agents}\n"
    s += f"    n_agent_types = 1\n"
    s += f"    n_pois = {n_pois}\n"
    s += f"    poi_options = {poi_options}\n"
    s += f"    with_agents = True\n"
    s += f"    size = 20\n"
    s += f"    time_steps = 60\n"
    s += f"    n_regions = 8\n"
    s += f"    sensor_range = 10\n"
    s += f"    rand_action_rate = 0.05\n\n"

    s += f"    # POI:\n"
    s += f"    value = 1\n"
    s += f"    obs_radius = 1\n"
    s += f"    couple = 1\n"
    s += f"    strong_coupling = False\n\n"

    s += f"    # Agent:\n"
    s += f"    capabilities = False\n\n"

    s += f"    # Neural Network:\n"
    s += f"    hid = 30\n\n"

    s += f"    # Evolve nn:\n"
    s += f"    sigma = 0.1\n"
    s += f"    learning_rate = 0.05\n"
    s += f"    n_policies = 50\n\n"

    s += f"    # Evo learner\n"
    s += f"    n_gen = 1500\n"
    return s

