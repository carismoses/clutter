import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pybullet as p

from agents.panda_agent import PandaAgent
from learning.domains.towers.generate_tower_training_data import sample_random_tower
from block_utils import Object, Dimensions, Position, Color, get_adversarial_blocks
from tamp.misc import load_blocks
import pb_robot

def main(args):
    NOISE=0.00005

    # Load blocks
    blocks = load_blocks(fname=args.blocks_file,
                         num_blocks=args.num_blocks)
    # blocks = get_adversarial_blocks(num_blocks=args.num_blocks)

    agent = PandaAgent(blocks, NOISE,
        use_planning_server=args.use_planning_server,
        alternate_orientations=args.alternate_orientations,
        use_vision=args.use_vision,
        real=args.real,
        task=args.task)

    if args.show_frames:
        agent.step_simulation(T=1, vis_frames=True, lifeTime=0.)
        input("Start building?")
        p.removeAllUserDebugItems()

    for tx in range(0, args.num_trials):
        print(f"\nStarting trial {tx}\n")
        if args.task == 'stacking':
            n_blocks = np.random.randint(2, min(args.num_blocks + 1, 5))
            tower_blocks = np.random.choice(blocks, n_blocks, replace=False)
            tower = sample_random_tower(tower_blocks)
            agent.simulate_tower(tower,
                                 real=args.real,
                                 base_xy=(0.5, -0.3),
                                 vis=True,
                                 T=2500)
        elif args.task == 'clutter':
            agent.simulate_clutter(real=args.real,
                                 base_xy=(0.5, -0.3),
                                 vis=True,
                                 T=2500)
        print(f"\nFinished trial {tx}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--task', default='clutter', choices=['clutter', 'stacking'], type=str)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--num-blocks', type=int, default=10)
    parser.add_argument('--num-trials', type=int, default=100)
    parser.add_argument('--save-tower', action='store_true')
    parser.add_argument('--use-planning-server', action='store_true')
    parser.add_argument('--alternate-orientations', action='store_true')
    parser.add_argument('--use-vision', action='store_true', help='get block poses from AR tags')
    parser.add_argument('--blocks-file', type=str, default='object_files/block_files/final_block_set_10.pkl')
    parser.add_argument('--real', action='store_true', help='run on real robot')
    parser.add_argument('--show-frames', action='store_true')
    args = parser.parse_args()
    if args.debug: pdb.set_trace()

    main(args)
