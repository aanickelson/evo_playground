from AIC.aic import aic
from evo_playground.support.get_policy_from_niche import PolicyMap
from evo_playground.support.rover_wrapper import RoverWrapper
import numpy as np

class TopPolEnv:
    def __init__(self, p, lp, pfile, cfile, bh_sz):
        self.env = aic(p)
        self.wrap = RoverWrapper(self.env)
        self.wrap.use_bh = False
        self.pmap = PolicyMap(pfile, cfile, bh_sz)
        self.p = p
        self.lp = lp
        self.pfile = pfile
        self.cfile = cfile
        self.moo_wts = np.array([0.5, 0.5])

    def G(self):
        return self.env.G()

    def D(self):
        return self.env.D()

    def run(self, models):
        low_level_pols = []
        for i, policy in enumerate(models):
            # Get the NN output (behavior)
            bh_choice = policy(self.moo_wts).detach().numpy()
            # Get a policy from a filled niche close to that behavior
            pol_choice = self.pmap.get_pol(bh_choice)
            if len(pol_choice) > 0:
                low_level_pols.append(pol_choice)
            else:
                return -3
        if len(low_level_pols) < len(models):
            print("something has gone wrong here")
            return -2

        G = np.array(self.wrap._evaluate(low_level_pols))
        return sum(self.moo_wts * G)

    def reset(self):
        self.env.reset()
