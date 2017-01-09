import argparse
import os
import sys
from surreal.utils.console import *
from surreal.utils.io import *

def a3c_tmux(session, num_workers, remotes, env_id, log_dir, shell='bash'):
    # for launching the TF workers and for launching tensorboard
    base_cmd = [
        'CUDA_VISIBLE_DEVICES=', sys.executable, f_join(parent_dir(__file__), 'worker.py'),
        '--log-dir', log_dir, '--env-id', env_id,
        '--num-workers', str(num_workers)]

    if remotes is None:
        remotes = ["1"] * num_workers
    else:
        remotes = remotes.split(',')
        assert len(remotes) == num_workers

    cmds_map = [new_tmux_cmd(session, "ps", base_cmd + ["--job-name", "ps"])]
    for i in range(num_workers):
        cmds_map += [new_tmux_cmd(session,
            "w-%d" % i, base_cmd + ["--job-name", "worker", "--task", str(i), "--remotes", remotes[i]])]

    cmds_map += [new_tmux_cmd(session, "tb", ["tensorboard --logdir {} --port 12345".format(log_dir)])]
    cmds_map += [new_tmux_cmd(session, "htop", ["htop"])]

    windows = [v[0] for v in cmds_map]

    cmds = [
        "mkdir -p {}".format(log_dir),
        "tmux kill-session -t {}".format(session),
        "tmux new-session -s {} -n {} -d {}".format(session, windows[0], shell)
    ]
    for w in windows[1:]:
        cmds += ["tmux new-window -t {} -n {} {}".format(session, w, shell)]
    cmds += ["sleep 1"]
    for window, cmd in cmds_map:
        cmds += [cmd]

    return cmds