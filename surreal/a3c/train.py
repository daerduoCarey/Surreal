import argparse
import os
import sys
from six.moves import shlex_quote
from surreal.utils.io.filesys import *
from .worker import SCRIPT_PATH

def new_cmd(session, name, cmd, mode, logdir, shell):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(shlex_quote(str(v)) for v in cmd)
    if mode == 'tmux':
        return "tmux send-keys -t {}:{} {} Enter".format(session, name, shlex_quote(cmd))
    elif mode == 'child':
        return "{} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh".format(cmd, logdir, session, name, logdir)
    elif mode == 'nohup':
        return "nohup {} -c {} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh".format(shell, shlex_quote(cmd), logdir, session, name, logdir)


def a3c_command(session, num_workers, remotes, env_id, logdir, shell='bash', mode='tmux', 
                port_offset=0, virtualenv_cmd=None, visualize=False, restart=False):
    # for launching the TF workers and for launching tensorboard
    # virtualenv_cmd: string like `source activate MyEnv` for tmux
    tb_port = str(10000 + port_offset * 10)
    cluster_port = str(15000 + port_offset * 20)

    base_cmd = [
        'CUDA_VISIBLE_DEVICES=',
        'python', SCRIPT_PATH,
        '--log-dir', logdir, 
        '--env-id', env_id,
        '--num-workers', str(num_workers)]

    if visualize:
        base_cmd += ['--visualize']

    if remotes is None:
        remotes = ["1"] * num_workers
    else:
        remotes = remotes.split(',')
        assert len(remotes) == num_workers

    windows = ['ps']
    cmds_map = [new_cmd(session, "ps", base_cmd + ["--job-name", "ps", "--port", cluster_port], mode, logdir, shell)]
    
    for i in range(num_workers):
        win = "w-{}".format(i)
        windows.append(win)
        cmds_map += [new_cmd(session, win, 
                             base_cmd + ["--job-name", "worker", "--task", str(i), "--remotes", remotes[i], "--port", cluster_port], mode, logdir, shell)]

    cmds_map += [new_cmd(session, "tb", ["tensorboard", "--logdir", logdir, "--port", tb_port], mode, logdir, shell)]
    if mode == 'tmux':
        cmds_map += [new_cmd(session, "htop", ["htop"], mode, logdir, shell)]
    windows += ['tb', 'htop']

    notes = []
    cmds = [
        "mkdir -p {}".format(logdir),
        "echo {} {} > {}/cmd.sh".format('python', ' '.join([shlex_quote(arg) for arg in sys.argv if arg != '-n']), logdir),
    ]
    if mode == 'nohup' or mode == 'child':
        cmds += ["echo '#!/bin/sh' >{}/kill.sh".format(logdir)]
        notes += ["Run `source {}/kill.sh` to kill the job".format(logdir)]
    if mode == 'tmux':
        notes += ["Use `tmux attach -t {}` to watch process output".format(session)]
        notes += ["Use `tmux kill-session -t {}` to kill the job".format(session)]
    else:
        notes += ["Use `tail -f {}/*.out` to watch process output".format(logdir)]
    notes += ["Point your browser to http://localhost:{} to see Tensorboard".format(tb_port)]

    if mode == 'tmux':
        cmds += [
        "tmux kill-session -t {}".format(session),
        'sleep 2',
        "rm -rf " + logdir if restart else '',
        "mkdir -p {}".format(logdir),
        "tmux new-session -s {} -n {} -d {}".format(session, windows[0], shell)
        ]
        for w in windows[1:]:
            cmds += ["tmux new-window -t {} -n {} {}".format(session, w, shell)]
        cmds += ["sleep 2"]
        if virtualenv_cmd:
            for w in windows:
                cmds.append(new_cmd(session, w, virtualenv_cmd, mode, logdir, shell))
                cmds.append('sleep 1')

    for cmd in cmds_map:
        cmds += [cmd]

    return cmds, notes


parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-w', '--num-workers', default=8, type=int,
                    help="Number of workers")
parser.add_argument('--remotes', default=None,
                    help='The address of pre-existing VNC servers and '
                         'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901).')
parser.add_argument('-e', '--env', type=str, default="Pong",
                    help="Environment short-name")
parser.add_argument('-s', '--suffix', type=str, default="",
                    help="Optional env suffix, will affect naming of logdir and tmux window")
parser.add_argument('-l', '--log-dir', type=str, default="~/Train",
                    help="Root log directory path")
parser.add_argument('-n', '--dry-run', action='store_true',
                    help="Print out commands rather than executing them")
parser.add_argument('-m', '--mode', type=str, default='tmux',
                    help="tmux: run workers in a tmux session. nohup: run workers with nohup. child: run workers as child processes")
parser.add_argument('-p', '--port-offset', type=int, default=1,
                    help='Port offset for Tensorboard and ClusterSpec')
parser.add_argument('-t', '--tmux-window', type=str, default='a3c',
                    help='Tmux window')
parser.add_argument('--visualize', action='store_true',
                    help="Visualize the gym environment by running env.render() between each timestep")
parser.add_argument('-r', '--restart', action='store_true',
                    help="Delete the old save dir of parameters, restart the training from scratch.")


def main():
    args = parser.parse_args()
    env = args.env
    assert 'Deterministic' not in env and '-v' not in env, 'only provide the main part'
    if args.suffix:
        log_id = '{}-{}'.format(env, args.suffix)
    else:
        log_id = env
    logdir = f_expand(f_join(args.log_dir, log_id))
    args.tmux_window = log_id
    
    ENV_SUFFIX = 'NoFrameskip-v3' if 1 else 'Deterministic-v3'
    
    cmds, notes = a3c_command(session=args.tmux_window, 
                              num_workers=args.num_workers, 
                              remotes=args.remotes, 
                              env_id=env + ENV_SUFFIX,
                              logdir=logdir, 
                              mode=args.mode,
                              port_offset=args.port_offset,
                              virtualenv_cmd='source activate bitworld',
                              visualize=args.visualize,
                              restart=args.restart)
    if args.dry_run:
        print("Dry-run mode due to -n flag, otherwise the following commands would be executed:")
    else:
        print("Executing the following commands:")
    print("\n".join(cmds))
    print("")
    if not args.dry_run:
        if args.mode == "tmux":
            os.environ["TMUX"] = ""
        os.system("\n".join(cmds))
    print('\n'.join(notes))


if __name__ == "__main__":
    main()