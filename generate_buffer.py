import gym
import numpy as np
import tensorflow as tf
import argparse
import os

import utils
import DDPG

# modified from BCQ in PyTorch code: https://github.com/sfujim/BCQ

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_name", default='LunarLanderContinuous-v2')				# OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)					# Sets Gym, TensorFlow and Numpy seeds
	parser.add_argument("--buffer_size", default=1e5, type=float)		# Max time steps to run environment for
	parser.add_argument("--noise1", default=0.3, type=float)			# Probability of selecting random action
	parser.add_argument("--noise2", default=0.3, type=float)			# Std of Gaussian exploration noise
	parser.add_argument("--actor_lr", default=0.0001, type=float)
	parser.add_argument("--critic_lr", default=0.001, type=float)
	parser.add_argument("--actor_hs", default=0, type=int)
	parser.add_argument("--critic_hs", default=0, type=int)
	parser.add_argument("--batch_size", default=100, type=int)
	parser.add_argument("--discount", default=0.99, type=float)
	parser.add_argument("--tau", default=0.005, type=float)
	parser.add_argument("--dqda_clip", default=None, type=float)
	parser.add_argument("--clip_norm", default=0, type=int)
	args = parser.parse_args()

	if args.actor_hs <= 0:
		actor_hs_list = [400,300]
	else:
		actor_hs_list = [args.actor_hs]*2
	if args.critic_hs <= 0:
		critic_hs_list = [400,300]
	else:
		critic_hs_list = [args.critic_hs] * 2

	file_name = "DDPG_%s_%s" % (args.env_name, str(args.seed))
	buffer_name = "Robust_%s_%s" % (args.env_name, str(args.seed))
	print("---------------------------------------")
	print("Settings: " + file_name)
	print("---------------------------------------")

	if not os.path.exists("./buffers"):
		os.makedirs("./buffers")

	env = gym.make(args.env_name)

	env.seed(args.seed)
	tf.reset_default_graph()
	tf.set_random_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = int(env.action_space.high[0])

	with tf.Session() as sess:
		# Initialize and load policy
		policy = DDPG.DDPG(state_dim, action_dim, max_action, sess, args.tau, actor_hs_list, args.actor_lr, critic_hs_list, args.critic_lr,
						   args.dqda_clip, bool(args.clip_norm) )
		policy.load(file_name, "./models")

		# Initialize buffer
		replay_buffer = utils.ReplayBuffer()

		total_timesteps = 0
		episode_num = 0
		done = True

		while total_timesteps < args.buffer_size:

			if done:

				if total_timesteps != 0:
					print("Total T: {} Episode Num: {} Episode T: {} Reward: {:.1f}"
						.format(total_timesteps, episode_num, episode_timesteps, episode_reward))

				# Reset environment
				obs = env.reset()
				done = False
				episode_reward = 0
				episode_timesteps = 0
				episode_num += 1

			# Add noise to actions
			if np.random.uniform(0, 1) < args.noise1:
				action = env.action_space.sample()
			else:
				action = policy.select_action(np.array(obs))
				if args.noise2 != 0:
					action = (action + np.random.normal(0, args.noise2, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

			# Perform action
			new_obs, reward, done, _ = env.step(action)
			done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
			episode_reward += reward

			# Store data in replay buffer
			replay_buffer.add((obs, new_obs, action, reward, 1-done_bool))

			obs = new_obs

			episode_timesteps += 1
			total_timesteps += 1

		# Save final buffer
		replay_buffer.save(buffer_name)