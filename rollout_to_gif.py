import imageio.v2 as imageio
import evo_playground.parameters as param
from os import getcwd, path

for p in param.BIG_BATCH_01:  # param.p328, param.p329, param.p402, param.p403]:  #  , param.p328, param.p329]:
    for pre in ['G_time', 'D_time']:
        # for trial in range(3):
        p.fname_prepend = pre
        print(p.trial_num)
        images = []
        paths = []
        for i in range(p.time_steps):
            paths.append(path.join(getcwd(), 'rollouts', 'rollouts', p.fname_prepend + "t{}_{}.png".format(p.trial_num, i)))

        for filename in paths:
            images.append(imageio.imread(filename))
        pth = path.join(getcwd(), 'gifs', p.fname_prepend + 'vis{}.gif'.format(p.trial_num))
        imageio.mimsave(pth, images)