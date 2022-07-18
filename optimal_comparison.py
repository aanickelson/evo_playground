from math import sqrt
import numpy as np
from evo_playground.parameters import BATCH2
from teaming.domain import DiscreteRoverDomain as Domain


def optimal_policy(env):
    for t in range(env.time_steps):
        acts = []
        # For each agent
        for a in env.agents:
            min_dist = 1000
            poi = False
            # Find the nearest unobserved POI not already claimed
            for i, p in enumerate(env.pois):
                if p.claimed or p.observed:
                    continue
                d = sqrt((a.x - p.x)**2 + (a.y - p.y)**2)
                if d < min_dist:
                    min_dist = d
                    poi = p
            # Collect actions for all agents
            acts.append(poi)
            if poi:
                poi.claimed = True
        # Take one step toward chosen POIs
        env.step(acts)
        # Reset POI choices
        for poi_n in env.pois:
            poi_n.claimed = False

    return env.G()


if __name__ == '__main__':
    for p in BATCH2:
        print("TRIAL {}".format(p.trial_num))
        captured = np.zeros(100)
        for j in range(100):
            env = Domain(p)
            captured[j] = optimal_policy(env) / env.theoretical_max_g
        print(np.mean(captured))
