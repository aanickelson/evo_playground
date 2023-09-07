import numpy as np
from AIC.aic import aic


class AICWrapper(aic):
    def __init__(self, p):
        super().__init__(p)

    def run(self, pols, use_bh=False):
        return run_env(self, pols, self.params, use_bh)


def run_env(env, policies, p, use_bh=False):
    bh_space = [[] for _ in range(p.n_agents)]
    n_eff_vals = 2  # number of values agent can output to determine velocity / effort
    for i in range(p.time_steps):
        state = env.state()
        actions = []
        for i, policy in enumerate(policies):
            action = policy(state[i]).detach().numpy()
            actions.append(action)

            if use_bh:
                bh_space[i].append(action_space(action, p.n_sensors, n_eff_vals))

        env.action(actions)

    if not use_bh:
        return env.G()
    else:
        return env.G(), calc_bh(bh_space, p.n_poi_types, p.n_agents, n_eff_vals)


def run(env, policies, p, use_bh=False):
    return run_env(env, policies, p, use_bh)


def action_space(act_vec, n_sensors, n_effort_vals):
    idx = np.argmax(act_vec[:-n_effort_vals])
    poi_type = int(np.floor(idx / n_sensors))
    return np.concatenate(([poi_type], act_vec[-n_effort_vals:]))


def calc_bh(bh_vec, n_poi_types, n_agents, n_beh):
    bh = np.zeros((n_agents, n_poi_types, n_beh))
    for ag in range(n_agents):
        ag_bhs = np.array(bh_vec[ag])

        for poi in range(n_poi_types):
            try:
                all_poi_bhs = ag_bhs[ag_bhs[:, 0] == poi]
                if all_poi_bhs.size > 0:
                    poi_bh = all_poi_bhs.mean(axis=0)[1:]
                    bh[ag][poi] = poi_bh
            except RuntimeWarning:
                pass

    bh = np.nan_to_num(bh, np.nan)
    bh = np.reshape(bh, (n_agents, n_poi_types * n_beh))
    return bh
