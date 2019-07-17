import numpy as np
import tensorflow as tf
import utils
import trfl


# modified from BCQ in PyTorch code: https://github.com/sfujim/BCQ

class BCQNetwork:
	def __init__(self, name, state_dim, action_dim, max_action, actor_hs_list, actor_lr, critic_hs_list, critic_lr, dqda_clipping=None, clip_norm=False):
		self.name = name
		with tf.variable_scope(self.name):
			# placeholders for actor and critic networks
			self.state_ = tf.placeholder(tf.float32, [None, state_dim], name='state')
			# placeholders for actor network
			self.action_ = tf.placeholder(dtype=tf.float32, shape=[None, action_dim], name='action')
			# placeholders for target critic soft q network
			self.reward_ = tf.placeholder(tf.float32, [None], name='rewards')
			self.discount_ = tf.placeholder(tf.float32, [None], name='discounts')
			self.flipped_done_ = tf.placeholder(tf.float32, [None], name='flipped_dones')
			# placeholders for training critic network
			self.target_ = tf.placeholder(tf.float32, [None, 1], name='target')

			# actor network
			# input is concat of state and action (in DDPG input is just the state)
			self.fc1_actor_ = tf.contrib.layers.fully_connected(tf.concat([self.state_, self.action_], axis=1), actor_hs_list[0], activation_fn=tf.nn.relu) #400
			self.fc2_actor_ = tf.contrib.layers.fully_connected(self.fc1_actor_, actor_hs_list[1], activation_fn=tf.nn.relu)
			self.fc3_actor_ = tf.contrib.layers.fully_connected(self.fc2_actor_, action_dim, activation_fn=tf.nn.tanh) * 0.05 * max_action
			self.actor_clip_ = tf.clip_by_value((self.fc3_actor_ + self.action_),-max_action, max_action)

			# critic network
			# double headed and combining state and actor actions on first level
			self.fc1_critic_1_ = tf.contrib.layers.fully_connected(tf.concat([self.state_, self.actor_clip_], axis=1),
																 critic_hs_list[0], activation_fn=tf.nn.relu)
			self.fc2_critic_1_ = tf.contrib.layers.fully_connected(self.fc1_critic_1_, critic_hs_list[1], activation_fn=tf.nn.relu)
			self.critic_1_out_ = tf.contrib.layers.fully_connected(self.fc2_critic_1_, 1, activation_fn=None)

			self.fc1_critic_2_ = tf.contrib.layers.fully_connected(tf.concat([self.state_, self.actor_clip_], axis=1),
																  critic_hs_list[0], activation_fn=tf.nn.relu)
			self.fc2_critic_2_ = tf.contrib.layers.fully_connected(self.fc1_critic_2_, critic_hs_list[1],
																  activation_fn=tf.nn.relu)
			self.critic_2_out_ = tf.contrib.layers.fully_connected(self.fc2_critic_2_, 1, activation_fn=None)

			# TRFL usage: deterministic policy gradients. In BCQ/TD3 arbitrarily choose first net work for the DPG
			self.dpg_return_ = trfl.dpg(self.critic_1_out_, self.actor_clip_, dqda_clipping=dqda_clipping, clip_norm=clip_norm)

			# train actor with DPG
			self.actor_loss_ = tf.reduce_mean(self.dpg_return_.loss)
			self.actor_optim_ = tf.train.AdamOptimizer(learning_rate=actor_lr).minimize(self.actor_loss_)

			# Soft Clipped Double Q-learning for target network value
			self.soft_q_ = 0.75 * tf.minimum(self.critic_1_out_, self.critic_2_out_) + 0.25 * tf.maximum(self.critic_1_out_, self.critic_2_out_)

			# soft_q has 10 x batch_size rows, take max of each of the repeated rows for the target_q
				# reshape (10*batch_size)x1 into batch_size x 10, then take max of the columns, reshape into batch_size x 1
			self.soft_q_max_ = tf.reshape(tf.reduce_max(tf.reshape(self.soft_q_,[-1,10]),axis=1),[-1,])

			self.target_q_ = self.reward_ + self.flipped_done_ * self.discount_ *self.soft_q_max_ #next_state estimate is 0 if done

			#train critic with combined losses of both critic network heads
			self.critic_loss_ = tf.losses.mean_squared_error(self.target_, self.critic_1_out_) \
								+ tf.losses.mean_squared_error(self.target_, self.critic_2_out_)
			self.critic_optim_ = tf.train.AdamOptimizer(learning_rate=critic_lr).minimize(self.critic_loss_)


	# get variables for actor and critic networks for target network updating
	def get_network_variables(self):
		return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]


# Vanilla Variational Auto-Encoder 
class VAE(object):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, vae_lr):
		# placeholders
		self.action_ = tf.placeholder(dtype=tf.float32, shape=[None, action_dim])
		self.state_ = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
		self.decoder_state_ = tf.placeholder(dtype=tf.float32, shape=[None, state_dim]) # can be same input as state depending on call
		self.random_batch_size_ = tf.placeholder(dtype=tf.int32, shape=[])
		# encoder 
		self.e1_ = tf.contrib.layers.fully_connected(tf.concat([self.state_, self.action_], axis=1), 750, activation_fn=tf.nn.relu)
		self.e2_ = tf.contrib.layers.fully_connected(self.e1_, 750, activation_fn=tf.nn.relu)
		# mean, std, z
		self.mean_ = tf.contrib.layers.fully_connected(self.e2_, latent_dim, activation_fn=None)
		self.log_std_ = tf.clip_by_value(tf.contrib.layers.fully_connected(self.e2_, latent_dim, activation_fn=None), -4, 15)
		self.std_ = tf.exp(self.log_std_)
		self.epsilon_ = tf.random_normal([self.random_batch_size_, latent_dim], mean= 0.0, stddev=1.0)
		self.z_ = self.mean_ + self.std_ * self.epsilon_
		# decoder
		self.d1_ = tf.contrib.layers.fully_connected(tf.concat([self.decoder_state_, self.z_],axis=1), 750, activation_fn=tf.nn.relu)
		self.d2_ = tf.contrib.layers.fully_connected(self.d1_, 750, activation_fn=tf.nn.relu)
		self.d3_ = tf.contrib.layers.fully_connected(self.d2_, action_dim, activation_fn=tf.nn.tanh) * max_action

		self.recon_loss_ = tf.losses.mean_squared_error(self.d3_, self.action_)
		self.KL_loss_ = -0.5 * tf.reduce_mean((1. + tf.log(tf.pow(self.std_,2)) - tf.pow(self.mean_,2)  - tf.pow(self.std_,2)))
		self.loss_ = self.recon_loss_ + 0.5 * self.KL_loss_
		self.optim_ = tf.train.AdamOptimizer(learning_rate=vae_lr).minimize(self.loss_)



class BCQ(object):
	def __init__(self, state_dim, action_dim, max_action, sess, tau=0.001, actor_hs=[400, 300], actor_lr=0.001,
				 critic_hs=[400, 300], critic_lr=0.001, dqda_clipping=None, clip_norm=False, vae_lr=0.001):

		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.latent_dim = action_dim * 2

		self.bcq_train = BCQNetwork("train_bcq", state_dim=state_dim, action_dim=action_dim, max_action=max_action,
									actor_hs_list=actor_hs, actor_lr=actor_lr, critic_hs_list=critic_hs, critic_lr=critic_lr,
									dqda_clipping=dqda_clipping, clip_norm=clip_norm)
		self.bcq_target = BCQNetwork("target_bcq", state_dim=state_dim, action_dim=action_dim, max_action=max_action,
									actor_hs_list=actor_hs, actor_lr=actor_lr, critic_hs_list=critic_hs, critic_lr=critic_lr,
									dqda_clipping=dqda_clipping, clip_norm=clip_norm)
		self.vae = VAE(state_dim, action_dim, self.latent_dim, max_action, vae_lr)

		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()
		# target network update operations
		self.target_network_update_op = trfl.update_target_variables(self.bcq_target.get_network_variables(),
																self.bcq_train.get_network_variables(), tau=tau)
		# intialize networks to start with the same variables:
		self.target_same_init = trfl.update_target_variables(self.bcq_target.get_network_variables(),
																self.bcq_train.get_network_variables(), tau=1.0)
		self.sess.run(self.target_same_init)


	def select_action(self, state):
		# duplicate state times 10
		state_duplicate = 10
		z = np.random.normal(0.,1.,size=(state_duplicate, self.latent_dim)).clip(-0.5,0.5)
		tile_state = np.tile(state,(state_duplicate,1))
		# get action from actor net w/ VAE generated action
		vae_action = self.sess.run(self.vae.d3_, feed_dict={
			self.vae.decoder_state_:tile_state,
			self.vae.z_:z,
			self.vae.random_batch_size_:state_duplicate
		})
		q1, bcq_action = self.sess.run([self.bcq_train.critic_1_out_, self.bcq_train.actor_clip_], feed_dict={
			self.bcq_train.state_:tile_state,
			self.bcq_train.action_:vae_action
		})
		action_index = np.argmax(q1)

		return bcq_action[action_index]


	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99):
		discount_batch = np.array([discount] * batch_size)
		stats_loss = {}
		stats_loss["actor_loss"] = 0.0
		stats_loss["critic_loss"] = 0.0
		stats_loss["vae_loss"] = 0.0
		for it in range(iterations):
			# Sample batches: done_batch has bool flipped in RL loop
			state_batch, next_state_batch, action_batch, reward_batch, flipped_done_batch = replay_buffer.sample(batch_size) #already flipped done bools

			# Variational Auto-Encoder Training
			vae_loss, _ = self.sess.run([self.vae.loss_, self.vae.optim_],feed_dict={
				self.vae.state_:state_batch,
				self.vae.action_:action_batch,
				self.vae.decoder_state_:state_batch,
				self.vae.random_batch_size_: batch_size
			})

			# Critic Training
			# Duplicate state 10 times
			state_rep = np.repeat(next_state_batch, 10, axis=0) # repeat each row by 10 in place (ie first 10 rows are row 0 repeated 10 times)
			# Compute value of perturbed actions sampled from the VAE
			z = np.random.normal(0., 1., size=(batch_size*10, self.latent_dim)).clip(-0.5, 0.5)
			vae_action = self.sess.run(self.vae.d3_, feed_dict={
				self.vae.decoder_state_: state_rep,
				self.vae.z_: z,
				self.vae.random_batch_size_: batch_size*10
			})
			# Get Soft Clipped Double Q-learning from target network
			target_q = self.sess.run(self.bcq_target.target_q_, feed_dict={
				self.bcq_target.state_:state_rep,
				self.bcq_target.action_:vae_action,
				self.bcq_target.reward_:reward_batch,
				self.bcq_target.flipped_done_:flipped_done_batch,
				self.bcq_target.discount_:discount_batch
			})

			# train bcq network
			critic_loss, _ = self.sess.run([self.bcq_train.critic_loss_, self.bcq_train.critic_optim_], feed_dict={
				self.bcq_train.state_:state_batch,
				self.bcq_train.action_:action_batch,
				self.bcq_train.target_:np.reshape(target_q,(-1,1))
			})

			# Pertubation Model / Action Training
			z = np.random.normal(0., 1., size=(batch_size, self.latent_dim)).clip(-0.5, 0.5)
			sampled_action = self.sess.run(self.vae.d3_, feed_dict={
				self.vae.decoder_state_: state_batch,
				self.vae.z_: z,
				self.vae.random_batch_size_: batch_size
			})
			perturbed_action = self.sess.run(self.bcq_train.actor_clip_, feed_dict={
				self.bcq_train.state_:state_batch,
				self.bcq_train.action_:sampled_action,
			})
			# Update actor through DPG
			actor_loss, _ = self.sess.run([self.bcq_train.actor_loss_, self.bcq_train.actor_optim_], feed_dict={
				self.bcq_train.state_:state_batch,
				self.bcq_train.action_:perturbed_action
			})

			# Update Target Networks
			self.sess.run(self.target_network_update_op)
			# get loss for stats for inner iteration
			stats_loss["actor_loss"] += actor_loss
			stats_loss["critic_loss"] += critic_loss
			stats_loss["vae_loss"] += vae_loss

		# return stats_loss
		stats_loss["actor_loss"] /= iterations
		stats_loss["critic_loss"] /= iterations
		stats_loss["vae_loss"] /= iterations
		return stats_loss

	def save(self, filename, directory):
		self.saver.save(self.sess, "{}/{}.ckpt".format(directory,filename))

	def load(self, filename, directory):
		self.saver.restore(self.sess, "{}/{}.ckpt".format(directory,filename))