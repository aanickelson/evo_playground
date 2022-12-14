import imageio.v2 as imageio
import evo_playground.parameters as param
from os import getcwd, path

for p in [param.p03]:
    images = []
    filenames = []
    for pre in ['G', 'D']:
        for i in range(p.time_steps):
            filenames.append(f"t{p.trial_num:02d}_{pre}_{i}.png")

        for filename in filenames:
            images.append(imageio.imread(filename))
        pth = path.join(getcwd(), f'vis{p.trial_num}_{pre}.gif')
        imageio.mimsave(pth, images)
