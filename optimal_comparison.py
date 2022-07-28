from math import sqrt
import numpy as np
import evo_playground.parameters as params
from teaming.domain import DiscreteRoverDomain as Domain


def optimal_policy(env):
    for t in range(env.time_steps):
        acts = []
        # For each agent
        for a in env.agents:
            min_dist = 1000
            poi = False
            if a.poi:
                poi = a.poi
            else:
                # Find the nearest unobserved POI not already claimed
                for i, p in enumerate(env.pois):
                    if (p.claimed >= p.obs_required) or p.observed or not p.active:
                        continue
                    d = sqrt((a.x - p.x)**2 + (a.y - p.y)**2)
                    if d < min_dist:
                        min_dist = d
                        poi = p
            # Collect actions for all agents
            acts.append(poi)
            if poi:
                poi.claimed += 1
        # Take one step toward chosen POIs
        env.step(acts)
        # Reset POI choices
        # for poi_n in env.pois:
        #     poi_n.claimed = False
        # env.draw(t)
    print(env.D())
    return env.G()


if __name__ == '__main__':
    for p in params.TEST_BATCH:
        print("TRIAL {}".format(p.trial_num))
        captured = np.zeros(100)

        env = Domain(p)
        env.visualize = True
        G = optimal_policy(env)
        print(G)
