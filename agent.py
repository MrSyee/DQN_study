import numpy as np
import tensorflow as tf
import os
import cv2
import gym

from model import DQN

if not os.path.isdir('./save'):
    os.mkdir('./save')

class Agent:
    def __init__(self, n_action, is_render=True):
        self.sess = tf.Session()

        self.model = DQN(self.sess, n_action)
        self.model_name = "DQN"

        self.env = gym.make("BreakoutDeterministic-v4")
        self.is_render = is_render

        self.EPISODE = 20

        # epsilon parameter
        self.epsilon_s = 1.0
        self.epsilon_e = 0.1
        self.epsilon_decay = 1000000
        self.epsilon = self.epsilon_s

        # train parameter
        self.train_start = 2
        self.update_target_rate = 1000

        self.n_action = n_action
        self.loss = 0

        # info
        self.total_q_max, self.total_loss = 0., 0.

        # tensor board
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter('summary/{}'.format(self.model_name), self.sess.graph)

        # save parameter
        self.save_episode_rate = 10


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


    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            q_value = self.sess.run(self.model.main_q_value, feed_dict={self.model.input_M_Q: state})
            action = np.max(q_value, 1)
            
        # decay epsilon
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
        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        obs = self.env.reset()
        print(self.env.action_space.n)
        print(self.env.unwrapped.get_action_meanings())
        print (np.shape(obs))
        obs = self.preprocessing(obs)
        state = np.concatenate((obs,obs,obs,obs), axis=3)

        global_step = 0
        for e in range(self.EPISODE):
            is_terminal = False
            step = 0
            total_reward = 0
            while(not is_terminal):
                global_step += 1
                step += 1

                action = self.get_action(state)
                observation, reward, is_terminal, info = self.env.step(action)
                if self.is_render:
                    self.env.render()
                observation = np.reshape(self.preprocessing(observation), [1, 84, 84, 1])
                next_state = np.append(observation, state[:,:,:,:3], axis=3)

                transition = [state, action, reward, next_state, is_terminal]
                self.model.replay_buffer.add_sample(transition)

                total_reward += reward
                self.total_q_max += self.sess.run(self.model.main_q_value, feed_dict={self.model.input_M_Q: state})
                state = next_state

                if self.model.replay_buffer.get_size() > self.train_start:
                    self.loss = self.model.train()
                    self.total_loss += self.loss

                if global_step % self.update_target_rate == 0:
                    self.model.update_target()


                print("global_step: {}  step: {}  loss: {:.4f} ".format(global_step, step, self.loss))

                if is_terminal == True:
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

                    print ("terminal state!!")

                if e % self.save_episode_rate == 0:
                    saver.save(self.sess, "./save/{}.ckpt".format(self.model_name))


def main(_):
    agent = Agent(n_action=4)
    agent.train()

if __name__=="__main__":
    tf.app.run()