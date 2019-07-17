import numpy as np
import tensorflow as tf
import utils
import trfl

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class DDPGNetwork:
	def __init__(self, name, state_dim, action_dim, max_action, actor_hs_list, actor_lr, critic_hs_list, critic_lr, dqda_clipping=None, clip_norm=False):
		self.name = name
		with tf.variable_scope(self.name):
			# placeholders for actor and critic networks
			self.input_ = tf.placeholder(tf.float32, [None, state_dim], name='inputs')
			# placeholders for training critic network
			self.reward_ = tf.placeholder(tf.float32, [None], name='rewards')
			self.discount_ = tf.placeholder(tf.float32, [None], name='discounts')
			self.target_ = tf.placeholder(tf.float32, [None], name='target')

			# actor network
			self.fc1_actor_ = tf.contrib.layers.fully_connected(self.input_, actor_hs_list[0], activation_fn=tf.nn.relu) #400
			self.fc2_actor_ = tf.contrib.layers.fully_connected(self.fc1_actor_, actor_hs_list[1], activation_fn=tf.nn.relu)
			self.fc3_actor_ = tf.contrib.layers.fully_connected(self.fc2_actor_, action_dim, activation_fn=tf.nn.tanh)
			self.actor_out_ = self.fc3_actor_ * max_action

			# critic network
			self.fc1_critic_ = tf.contrib.layers.fully_connected(self.input_, critic_hs_list[0], activation_fn=tf.nn.relu)
			self.fc2_critic_ = tf.contrib.layers.fully_connected(tf.concat([self.fc1_critic_, self.actor_out_], axis=1),
																 critic_hs_list[1], activation_fn=tf.nn.relu)
			self.critic_out_ = tf.contrib.layers.fully_connected(self.fc2_critic_, 1, activation_fn=None)

			# TRFL usage: deterministic policy gradients
			self.dpg_return_ = trfl.dpg(self.critic_out_, self.actor_out_, dqda_clipping=dqda_clipping, clip_norm=clip_norm)

			# train actor with DPG
			self.actor_loss_ = tf.reduce_mean(self.dpg_return_.loss)
			self.actor_optim_ = tf.train.AdamOptimizer(learning_rate=actor_lr).minimize(self.actor_loss_)

			# train critic with TRFL td_learning. target is critic target network output of next state
			self.td_return_ = trfl.td_learning(tf.squeeze(self.critic_out_), self.reward_, self.discount_, self.target_)
			self.critic_loss_ = tf.reduce_mean(self.td_return_.loss)
			self.critic_optim_ = tf.train.AdamOptimizer(learning_rate=critic_lr).minimize(self.critic_loss_)


	# get variables for actor and critic networks for target network updating
	def get_network_variables(self):
		return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]


class DDPG:
	def __init__(self, state_dim, action_dim, max_action, sess, tau = 0.001, actor_hs=[32,32], actor_lr=0.001, critic_hs=[32,32], critic_lr=0.001,
				 dqda_clipping=None, clip_norm=False):
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.max_action = max_action
		self.actor_hs_list = actor_hs
		self.actor_lr = actor_lr
		self.critic_hs_list = critic_hs
		self.critic_lr = critic_lr
		self.dqda_clipping = dqda_clipping
		self.clip_norm = clip_norm

		# create policy networks
		self.ddpg_net = DDPGNetwork("ddpg_train", self.state_dim, self.action_dim, self.max_action, self.actor_hs_list,
									self.actor_lr, self.critic_hs_list, self.critic_lr, self.dqda_clipping, self.clip_norm)
		self.ddpg_target = DDPGNetwork("ddpg_target", self.state_dim, self.action_dim, self.max_action, self.actor_hs_list,
									   self.actor_lr, self.critic_hs_list, self.critic_lr, self.dqda_clipping, self.clip_norm)
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()
		# target network update operations
		self.target_network_update_op = trfl.update_target_variables(self.ddpg_target.get_network_variables(),
																self.ddpg_net.get_network_variables(), tau=tau)

	def select_action(self, state):
		action = self.sess.run(self.ddpg_net.actor_out_,feed_dict={self.ddpg_net.input_:np.expand_dims(state,axis=0)})
		action = action[0]
		return action

	def train(self, replay_buffer, iterations=500, batch_size=100, discount=0.99):
		discount_batch = np.array([discount] * batch_size)
		actor_loss_stat, critic_loss_stat = 0., 0.
		for it in range(iterations):
			# Sample batches: done_batch has bool flipped in RL loop
			state_batch, next_state_batch, action_batch, reward_batch, done_batch = replay_buffer.sample(batch_size)

			# critic target
			target_v = self.sess.run(self.ddpg_target.critic_out_, feed_dict={
				self.ddpg_target.input_: next_state_batch
			})
			# if done, no target value for next state
			target_v = np.reshape(target_v, (-1,))*done_batch #flip done bool when adding to replay buffer

			critic_loss, _ = self.sess.run([self.ddpg_net.critic_loss_, self.ddpg_net.critic_optim_], feed_dict={
				self.ddpg_net.input_: state_batch,
				self.ddpg_net.actor_out_: action_batch,
				self.ddpg_net.reward_: reward_batch,
				self.ddpg_net.discount_: discount_batch,
				self.ddpg_net.target_: target_v
			})

			# train actor
			actor_loss, _ = self.sess.run([self.ddpg_net.actor_loss_, self.ddpg_net.actor_optim_], feed_dict={
				self.ddpg_net.input_: state_batch,
			})

			# Update the frozen target models
			self.sess.run(self.target_network_update_op)
			actor_loss_stat += actor_loss
			critic_loss_stat += critic_loss

		return actor_loss_stat, critic_loss_stat

	def save(self, filename, directory):
		self.saver.save(self.sess, "{}/{}.ckpt".format(directory,filename))

	def load(self, filename, directory):
		self.saver.restore(self.sess, "{}/{}.ckpt".format(directory,filename))
