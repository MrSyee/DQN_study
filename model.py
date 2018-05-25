import tensorflow as tf
import numpy as np

from replay_buffer import Replay_Buffer

slim = tf.contrib.slim

class DQN:
    def __init__(self, session, num_action):
        self.sess = session

        self.n_action = num_action
        self.height = 84
        self.width = 84
        self.input_num = 4 # 몇 개의 image를 state로 보는지
        self.batch_size = 2

        self._build_input()
        self.main_q_value = self._build_network(self.input_M_Q, "main_q_net")
        self.target_q_value = self._build_network(self.input_T_Q, "target_q_net")

        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.loss_op, self.train_op = self._build_op()

        # replay buffer
        self.capacity = 40000
        self.replay_buffer = Replay_Buffer(self.capacity)

    def _build_input(self):
        # set placeholder
        self.input_M_Q = tf.placeholder(tf.float32, shape=(None, self.height, self.width, self.input_num))
        self.input_T_Q = tf.placeholder(tf.float32, shape=(None, self.height, self.width, self.input_num))
        # action placeholder used to get main Q(S,A)
        self.input_A = tf.placeholder(tf.int32, shape=(None))
        # target Y placehoder
        self.input_Y = tf.placeholder(tf.float32, shape=(None))

    def _build_network(self, inputs, name):
        '''
        q_network model
        :args:
            inputs : placeholder, state ( tf.float32, preprosessed img size x 4 )
            name: String, scope name 
        :return:
            logits : q_value, [action_size]
        '''
        with tf.variable_scope(name):
            conv1 = slim.conv2d(inputs, num_outputs=32, kernel_size=[8, 8], stride=[4, 4])
            conv2 = slim.conv2d(conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2])
            conv3 = slim.conv2d(conv2, num_outputs=64, kernel_size=[3, 3])
            conv3_flatten = slim.flatten(conv3)
            fc1 = slim.fully_connected(conv3_flatten, 512)
            q_value = slim.fully_connected(fc1, self.n_action, activation_fn=None)

        return q_value


    def _build_op(self):
        # get main Q with A, Q(S,A): [batch_size,]
        main_Q_with_A = self._get_Q_with_A(self.main_q_value)
        loss_op = tf.reduce_mean(tf.square(self.input_Y - main_Q_with_A))
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op)
        return loss_op, train_op


    def _get_Q_with_A(self, Q_value):
        # get Q(S,A): [batch_size,]
        one_hot = tf.one_hot(self.input_A, self.n_action, 1.0, 0.0)
        Q_with_A = tf.reduce_sum(tf.multiply(Q_value, one_hot), axis=1)

        return Q_with_A


    def _get_Y(self, next_state, reward):
        '''
        # get Y
        :param
            next_state: next_state, [1, height, width, num_img]
            reward: reward, [1]
        :return: Y, [1]
        '''
        next_state = np.expand_dims(next_state, axis=0)
        argmaxQ = self.sess.run(tf.argmax(self.target_q_value, axis=1),
                                feed_dict={self.input_T_Q: next_state})
        Q_with_A = self._get_Q_with_A(self.target_q_value)
        target_Q = self.sess.run(Q_with_A,
                                feed_dict={self.input_T_Q: next_state,
                                           self.input_A: argmaxQ})
        Y = reward + self.discount_factor * target_Q

        return Y


    def train(self):
        '''
        # Perform a gradient descent step on (y_j-Q(ð_j,a_j;θ))^2
        '''
        # get transition from replay buffer [state, action, reward, next_state, is_terminal]
        transition = self.replay_buffer.sampling(self.batch_size)
        state = np.zeros([self.batch_size, self.height, self.width, self.input_num])
        next_state = np.zeros([self.batch_size, self.height, self.width, self.input_num])
        action, reward, is_terminal = [], [], []
        i = 0
        for t in transition:
            state[i] = t[0]
            action.append(t[1])
            reward.append(t[2])
            next_state[i] = t[3]
            is_terminal.append(t[4])
            i += 1

        # get target Y: [batch_size,]
        Y = []
        for i in range(self.batch_size):
            if is_terminal[i]:
                y = reward[i]
            else:
                y = self._get_Y(next_state[i], reward[i])
            Y.append(y)
        Y = np.squeeze(Y)

        loss, _ = self.sess.run([self.loss_op, self.train_op], feed_dict={self.input_M_Q: state,
                                                                self.input_A: action,
                                                                self.input_Y: Y})
        return loss


    def update_target_network(self):
        # copy weight from main_q_network to target_q_network
        copy_op = []

        main_q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main_q_net')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_q_net')

        # 학습 네트웍의 변수의 값들을 타겟 네트웍으로 복사해서 타겟 네트웍의 값들을 최신으로 업데이트합니다.
        for main_q_var, target_var in zip(main_q_vars, target_vars):
            copy_op.append(target_var.assign(main_q_var.value()))

        self.sess.run(copy_op)





class DDQN(DQN):
    def _get_Y(self, next_state, reward):
        # argmaxQ 를 main_q로 구함
        argmaxQ = self.sess.run(tf.argmax(self.main_q_value),
                                feed_dict={self.input_M_Q: next_state})

        Q_with_A = self._get_Q_with_A(self.target_q_value)
        target_Q = self.sess.run(Q_with_A,
                                feed_dict={self.input_T_Q: next_state,
                                           self.input_A: argmaxQ})
        Y = reward + self.discount_factor * target_Q

        return Y
