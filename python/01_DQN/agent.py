import numpy as np
import tensorflow as tf
import os
import time
import cv2
import gym
from datetime import datetime

from model import DQN
import wrappers

NOWTIME = datetime.now().strftime("%Y%m%d-%H%M%S")

if not os.path.isdir('./save'):
    os.mkdir('./save')


class Agent:
    def __init__(self, n_action, is_render=True, is_load=False):
        self.sess = tf.Session()

        self.batch_size = 32

        self.model = DQN(self.sess, n_action, self.batch_size)
        self.model_name = "DQN"

        self.env = wrappers.wrap_dqn(gym.make("BreakoutDeterministic-v4"))
        self.is_render = is_render

        self.EPISODE = 600

        # epsilon parameter
        self.epsilon_s = 1.0
        self.epsilon_e = 0.1
        self.epsilon_decay = 100000
        self.epsilon = self.epsilon_s

        # train parameter
        self.train_start = 5000
        self.update_target_rate = 5000

        self.n_action = n_action
        self.loss = 0

        # info
        self.total_q_max, self.total_loss = 0., 0.

        # save parameter
        self.save_episode_rate = 5

        # load parameter
        self.is_load = is_load
        # saved_model = "./save/{}/{}_episode20.ckpt-{}".format("20180613-132735", self.model_name, "3741")
        self.saved_model = tf.train.latest_checkpoint("./save/20180614-180138")

    def preprocessing(self, img):
        '''
        args :
        img : ( 210 x 160 x 3 )

        return :
        img : ( 1 x 84 x 84 x 1 )
        '''
        # RGB to gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # resize
        img = cv2.resize(img, (84, 84))

        # Normalization
        img = (img - 127.5) / 127.5

        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=3)

        return img

    def get_action(self, state, is_play=False):
        if is_play:
            self.epsilon = self.play_epsilon

        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            q_value = self.sess.run(self.model.main_q_value, feed_dict={self.model.input_M_Q: state})
            action = np.argmax(q_value, 1)[0]

        # decay epsilon
        if not is_play:
            self.epsilon -= (self.epsilon_s - self.epsilon_e)/self.epsilon_decay

        return action

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def train(self):
        # tensor board
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter('graphs/{}/{}'
                                                    .format(self.model_name, NOWTIME), self.sess.graph)

        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        if self.is_load:
            print(self.saved_model)
            saver.restore(self.sess, self.saved_model)

        print("Train Start...")
        global_step = 0
        for e in range(self.EPISODE):
            obs = self.env.reset()
            #obs = self.preprocessing(obs)
            '''
            print(self.env.action_space.n)
            print(self.env.unwrapped.get_action_meanings())
            print (np.shape(obs))
            '''
            obs = np.reshape(obs, [1, 84, 84, 1])
            state = np.concatenate((obs, obs, obs, obs), axis=3)

            is_terminal = False
            step = 0
            total_reward = 0
            s_t = time.time()
            while not is_terminal:
                global_step += 1
                step += 1

                action = self.get_action(state)
                observation, reward, is_terminal, info = self.env.step(action)

                if self.is_render:
                    self.env.render()
                observation = np.reshape(observation, [1, 84, 84, 1])
                next_state = np.append(observation, state[:,:,:,:3], axis=3)

                transition = [state, action, reward, next_state, is_terminal]
                self.model.replay_buffer.add_sample(transition)

                total_reward += reward
                self.total_q_max += np.argmax(self.sess.run(self.model.main_q_value,
                                                            feed_dict={self.model.input_M_Q: state}), 1)
                state = next_state

                if self.model.replay_buffer.get_size() > self.train_start:

                    self.loss = self.model.train()
                    self.total_loss += self.loss

                if global_step % self.update_target_rate == 0:
                    self.model.update_target_network()

                if global_step % 20 == 0:
                    print("Episode: {}   global_step: {}  step: {}  loss: {:.4f}  reward: {}  time: {}".
                          format(e+1, global_step, step, self.loss, total_reward, time.time() - s_t))
                if is_terminal:
                    # write tensorboard
                    if self.model.replay_buffer.get_size() > self.train_start:
                        avg_q_max = self.total_q_max / float(step)
                        avg_loss = self.total_loss / float(step)

                        stats = [total_reward, avg_q_max, step, avg_loss]

                        for i in range(len(stats)):
                            self.sess.run(self.update_ops[i], feed_dict={
                                self.summary_placeholders[i]: float(stats[i])})
                        summary_str = self.sess.run(self.summary_op)
                        self.summary_writer.add_summary(summary_str, e + 1)

                    print("Episode: {}   global_step: {}  step: {}  loss: {:.4f}  reward: {}  time: {}".
                          format(e+1, global_step, step, self.loss, total_reward, time.time() - s_t))

                    self.total_loss, self.total_q_max = 0, 0

                if e % self.save_episode_rate == 0:
                    saver.save(self.sess, "./save/{0}/{1}_episode{2}.ckpt".format(NOWTIME, self.model_name, e), global_step=global_step)

    def play(self):
        self.play_epsilon = 0.1

        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        print(self.saved_model)
        saver.restore(self.sess, self.saved_model)

        print("Play Start...")
        for e in range(1):
            obs = self.env.reset()
            obs = np.reshape(obs, [1, 84, 84, 1])
            #obs = self.preprocessing(obs)
            self.env.render()
            state = np.concatenate((obs, obs, obs, obs), axis=3)

            is_terminal = False
            step = 0
            total_reward = 0
            while not is_terminal:
                step += 1

                action = self.get_action(state, is_play=True)
                print("action: {}".format(action))
                observation, reward, is_terminal, info = self.env.step(action)
                self.env.render()
                observation = np.reshape(observation, [1, 84, 84, 1])
                next_state = np.append(observation, state[:,:,:,:3], axis=3)

                total_reward += reward
                state = next_state

            print("step: {}   total_reward: {}".format(step, total_reward))


def main(_):
    start_time = time.time()
    agent = Agent(n_action=4, is_render=False, is_load=False)
    agent.train()
    #agent.play()
    print("Total train time : {}".format(time.time() - start_time))


if __name__ == "__main__":
    tf.app.run()