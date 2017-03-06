from __future__ import print_function
from collections import namedtuple
import time
import numpy as np
import tensorflow as tf
from surreal.model.simple import *
import six.moves.queue as queue
import threading
import distutils.version
from surreal.utils.image import *
from surreal.utils.io.filesys import *
from surreal.utils.common import CheckInterval, CheckPeriodic


# ============ hypers ============
LR = 3e-4
TOTAL_STEPS = 80e6
OPTIMIZER = 'rms'
USE_GAE = True
LOCAL_STEPS = 5
# ================================


def discount_(rewards, gamma, dones=None):
    """
    discount([30,20,10], 0.8) -> [52.4, 28., 10.]
    [30 + 20*.8 + 10*.8^2, 28 + 10*.8, 10]
    One-liner:
    ```
    import scipy.signal
    scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    ```
    """
    discounted = []
    r = 0
    if dones is None:
        dones = [0] * len(rewards)
    for reward, done in zip(reversed(rewards), reversed(dones)):
        r = reward + gamma*r
        r = r * (1. - done)
        discounted.append(r)
    return np.array(discounted[::-1], dtype=np.float32)


import scipy.signal
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def process_rollout(rollout, gamma, lambda_=1.0):
    """
    given a rollout, compute its returns and the advantage
    """
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    if USE_GAE:
        # this formula for the advantage comes "Generalized Advantage Estimation":
        # https://arxiv.org/abs/1506.02438
        delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
        batch_adv = discount(delta_t, gamma * lambda_)
    else:
        batch_adv = batch_r - rollout.values
    features = rollout.features[0]
    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)

        

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])


class PartialRollout(object):
    """
    a piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []

    def add(self, state, action, reward, value, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)


def env_runner(env, policy, num_local_steps, summary_writer, render):
    last_state = env.reset()
    last_features = policy.get_initial_features()
    length = 0
    rewards = 0

    while True:
        terminal_end = False
        rollout = PartialRollout()

        for _ in range(num_local_steps):
            action, value, features = policy.act(last_state, *last_features)
            # argmax to convert from one-hot
            state, reward, terminal, info = env.step(action.argmax())
            
            # TEMP
#             import random
#             if random.randint(0, 10000) == 5:
#                 img = (state * 255.).astype(np.uint8)
#                 save_img(img, f_expand('~/Temp/imgs/{}.png'.format(random.randint(1, 1000000000000))))
            
            if render:
                env.render()

            # collect the experience
            rollout.add(last_state, action, reward, value, terminal, last_features)
            length += 1
            rewards += reward

            last_state = state
            last_features = features

            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if terminal or length >= timestep_limit:
                terminal_end = True
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                last_features = policy.get_initial_features()
                print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
                length = 0
                rewards = 0
                break

        if not terminal_end:
            # if a partial rollout, use value function at the last step to bootstrap
            rollout.r = policy.value(last_state, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout


class A3C(object):
    def __init__(self, env, task, policy_class=CNNPolicy, visualize=False, mode='train'):
        self.env = env
        self.task = task
        self.is_train = (mode == 'train')
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = policy_class(env.observation_space.shape, env.action_space.n, mode)
            # tf.train.Supervisor will display global_step on TB. Create a new varscope to workaround the naming.
            with tf.variable_scope("diagnostics"):
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = policy_class(env.observation_space.shape, env.action_space.n, mode)
                pi.global_step = self.global_step

            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")

            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)

            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            # TODO: use tf.nn.sparse....
            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # loss of value function
            # TODO: reduce_mean?
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            bs = tf.to_float(tf.shape(pi.x)[0])
            # DM trick: learning rate for critic is HALF that of actor
            self.loss = pi_loss + 0.5 * vf_loss - entropy * 0.01

            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
#             self.runner = RunnerThread(env, pi, 20, visualize)

            grads = tf.gradients(self.loss, pi.var_list)

            tf.summary.scalar("diagnostics/policy_loss", pi_loss / bs)
            tf.summary.scalar("diagnostics/value_loss", vf_loss / bs)
            tf.summary.scalar("diagnostics/entropy", entropy / bs)
            tf.summary.image("diagnostics/state", pi.x)
            tf.summary.scalar("diagnostics/grad_global_norm", tf.global_norm(grads))
            tf.summary.scalar("diagnostics/var_global_norm", tf.global_norm(pi.var_list))
            self.summary_op = tf.summary.merge_all()

            grads, _ = tf.clip_by_global_norm(grads, 5.0)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])
            lr = tf.train.polynomial_decay(LR, self.global_step, TOTAL_STEPS, 1e-9)

            if self.is_train:
                # each worker has a different set of adam optimizer parameters
                if OPTIMIZER.lower() == 'adam':
                    opt = tf.train.AdamOptimizer(lr)
                else:
                    opt = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=1e-5)
                self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            else:
                self.train_op = tf.no_op()
                
            self.summary_writer = None
            self.summary_period = CheckPeriodic(100)
            self.eval_interval = CheckInterval(20000)


    def start(self, sess, summary_writer):
#         self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer
        # don't limit eval mode
        num_local_steps=LOCAL_STEPS if self.is_train else 10000000
        self.env_runner = env_runner(self.env, 
                                     self.local_network, 
                                     num_local_steps=num_local_steps,
                                     summary_writer=self.summary_writer, 
                                     render=False)


    def pull_batch_from_queue(self):
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout


    def process(self, sess):
        """
        process grabs a rollout that's been produced by the thread runner,
        and updates the parameters.  The update is then sent to the parameter
        server.
        """
        # print('DEBUG', self.global_step.eval())
        if (self.is_train or 
            self.eval_interval.trigger(self.global_step.eval())):
            # in eval mode, only evaluate every N steps.
            sess.run(self.sync)  # copy weights from shared to local
            rollout = next(self.env_runner)
        
        if not self.is_train:
            time.sleep(10) # don't evaluate too often
            return

        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)
        should_compute_summary = self.task == 0 and self.summary_period.trigger()

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        feed_dict = {
            self.local_network.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
        }
        feed_dict = self.local_network.state_in_feed(batch.features[:2],
                                                     feed_dict)

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()