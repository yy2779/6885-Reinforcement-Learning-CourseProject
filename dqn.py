import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 fc_map,
                 learning_rate=0.01,
                 scale=0.1,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False):

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net(scale, fc_map)

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self, scale, fc_map):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s,
                                 fc_map[0],
                                 tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                 bias_initializer=tf.constant_initializer(0.01),
                                 kernel_regularizer=tf.contrib.layers.l1_regularizer(scale),
                                 name='e1')
            self.q_eval = tf.layers.dense(e1,
                                          self.n_actions,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                          bias_initializer=tf.constant_initializer(0.01),
                                          kernel_regularizer=tf.contrib.layers.l1_regularizer(scale),
                                          name='e2')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_,
                                 fc_map[0],
                                 tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                 bias_initializer=tf.constant_initializer(0.01),
                                 kernel_regularizer=tf.contrib.layers.l1_regularizer(scale),
                                 name='t1')
            self.q_next = tf.layers.dense(t1,
                                          self.n_actions,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                          bias_initializer=tf.constant_initializer(0.01),
                                          kernel_regularizer=tf.contrib.layers.l1_regularizer(scale),
                                          name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.lr, global_step, 1000, 0.96, staircase=True)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self._train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        ar = np.asarray((a, r)).reshape(-1,1).astype(np.float32)
        transition = np.concatenate((s, ar, s_), axis=0).reshape(-1)
        # transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, epsilon_=None, verbose=False):
        # to have batch dimension when feed into tf placeholder
        observation = np.asarray(observation).reshape(1,-1)

        if epsilon_ is not None:
            epsilon = epsilon_
        else:
            epsilon = self.epsilon

        if np.random.uniform() <= epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            if verbose:
                print(actions_value)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('target_params_replaced:  ', self.learn_step_counter)
            print("epsilon: ", self.epsilon)

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        samples = self.memory[sample_index, :]

        _, cost = self.sess.run([self._train_op, self.loss], feed_dict={self.s: samples[:, :self.n_features],
                                                                        self.a: samples[:, self.n_features],
                                                                        self.r: samples[:, self.n_features + 1],
                                                                        self.s_: samples[:, -self.n_features:]})

        # print(cost)
        self.cost_his.append(cost)

        # increasing epsilon
        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.epsilon_increment
        else:
            self.epsilon = self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        # num = 100
        # cost_history = np.zeros([len(self.cost_his)//num, ])
        # for i in range(len(self.cost_his)//num):
        #     cost_history[i] = np.average(self.cost_his[i*num:(i+1)*num])
        print(np.average(self.cost_his))
        plt.plot(self.cost_his, lw=0.2)
        # print(self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.axis([0, 80000, 0, 800])
        plt.show()
