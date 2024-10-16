import argparse
import os

from belief_learner.config.utils import toml_to_config
from pprint import pprint
from belief_learner.train import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", action='store_true')
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--timesteps", type=float, default=1e6)
parser.add_argument("--disable-early-stopping", default=False, action='store_true')
args = parser.parse_args()

if not args.cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

n_cpus = len(os.sched_getaffinity(0))
os.environ['OMP_NUM_THREADS'] = str(n_cpus)


if __name__ == '__main__':
    config = toml_to_config(args.config)
    pprint(config)

    trainer = Trainer(config,
                      experiment_path=f"{os.path.dirname(__file__)}/experiments",
                      disable_wandb=True,
                      disable_early_stopping=args.disable_early_stopping)
    trainer.train(int(args.timesteps), int(1e4))

