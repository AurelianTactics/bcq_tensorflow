import numpy as np

# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# modified from BCQ in PyTorch code: https://github.com/sfujim/BCQ

# Simple replay buffer
class ReplayBuffer:
	def __init__(self, max_size=1e6):
		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	# # Expects tuples of (state, next_state, action, reward, done)
	# def add(self, data):
	# 	self.storage.append(data)
	def add(self, data):
		if len(self.storage) == self.max_size:
			self.storage[int(self.ptr)] = data
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append(data)

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		state, next_state, action, reward, done = [], [], [], [], []

		for i in ind: 
			s, s2, a, r, d = self.storage[i]
			state.append(np.array(s, copy=False))
			next_state.append(np.array(s2, copy=False))
			action.append(np.array(a, copy=False))
			reward.append(np.array(r, copy=False))
			done.append(np.array(d, copy=False))

		return (np.array(state), 
			np.array(next_state), 
			np.array(action), 
			np.array(reward).reshape(-1,), #np.array(reward).reshape(-1, 1)
			np.array(done).reshape(-1,)) #np.array(done).reshape(-1, 1))

	def save(self, filename):
		np.save("./buffers/"+filename+".npy", self.storage)

	def load(self, filename):
		self.storage = np.load("./buffers/"+filename+".npy",allow_pickle=True)