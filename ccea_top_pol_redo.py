from AIC.aic import aic
from evo_playground.support.get_policy_from_niche import PolicyMap
from evo_playground.support.rover_wrapper import RoverWrapper
import numpy as np

class TopPolEnv:
    def __init__(self, p, lp, pfile, cfile, bh_sz, only_bh=False, only_obj=False):
        self.env = aic(p)
        self.wrap = RoverWrapper(self.env)
        self.wrap.use_bh = False
        self.select_only_bh = only_bh
        self.select_only_obj = only_obj
        self.pmap = PolicyMap(pfile, cfile, bh_sz)
        self.p = p
        self.lp = lp
        self.pfile = pfile
        self.cfile = cfile
        self.moo_wts = np.array([[0.5, 0.5]])

    def G(self):
        return self.env.G()

    def D(self):
        return self.env.D()

    def G_exp(self, gs, wts):
        def rem_div_zero(v, min_val=1e-20):
            vals = np.array(v)
            vals[vals < min_val] = min_val
            return vals

        b = rem_div_zero(wts)
        g = rem_div_zero(gs)
        # Get the angle of the preferred balance and of the g balance
        theta_wts = np.arctan(b[1] / b[0])
        theta_gs = np.arctan(g[1] / g[0])
        # Take the difference between the angles, then scale so closer to the angle has a higher value
        # -10 is a scalar you can play with. Higher scalar values give a steeper gradient near the ideal tradeoff
        exp_val = np.exp(-10 * abs(theta_wts - theta_gs))
        # Then scale so further from the origin has a higher value
        return exp_val * np.sqrt(g[0] ** 2 + g[1] ** 2)

    def run(self, models):
        total_G = 0
        for wt in self.moo_wts:
            low_level_pols = []
            for i, policy in enumerate(models):
                # Get the NN output (behavior)
                bh_choice = policy(wt).detach().numpy()
                # Get a policy from a filled niche close to that behavior
                pol_choice = self.pmap.get_pol(bh_choice, self.select_only_bh, self.select_only_obj)
                if len(pol_choice) > 0:
                    low_level_pols.append(pol_choice)
                else:
                    return -3
            if len(low_level_pols) < len(models):
                print("there are fewer policies than there should be")
                return -2

            G = np.array(self.wrap._evaluate(low_level_pols))
            scalar_G = self.G_exp(G, wt)
            total_G += scalar_G
        total_G = total_G / len(self.moo_wts)
        return total_G

    def reset(self):
        self.env.reset()
