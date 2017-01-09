import os
import argparse
from surreal.a3c.train import a3c_tmux
from surreal.utils.console import *

parser = argparse.ArgumentParser(description="A3C training")
parser.add_argument('-w', '--num-workers', default=1, type=int,
                    help="Number of workers")
parser.add_argument('-r', '--remotes', default=None,
                    help='The address of pre-existing VNC servers and '
                         'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901).')
parser.add_argument('-e', '--env-id', type=str, default="PongDeterministic-v3",
                    help="Environment id")
parser.add_argument('-l', '--log-dir', type=str, default="/tmp/pong",
                    help="Log directory path")
parser.add_argument('-d', '--dry-run', action='store_true',
                    help="Print out the commands without actually running.")

args = vars(parser.parse_args())
dry_run = args.pop('dry_run')
cmds = a3c_tmux('a3c', **args)
cmd = '\n'.join(cmds)

print(cmd)
if dry_run:
    print('='*30, 'Dry run', '=' * 30)
else:
    os.system(cmd)