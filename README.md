# BCQ in TensorFlow

Authors' PyTorch BCQ implementation: https://github.com/sfujim/BCQ of BCQ paper https://arxiv.org/abs/1812.02900

utils.py: Contains replay buffer.
train_expert.py: Call this file to train a DDPG agent using DDPG.py.
generate_buffer.py: Call this file to generate a replay buffer using an agent (for example one created using train_expert.py)
main.py: Call this file to train a BCQ agent. Requires a replay buffer to be loaded (like one generated from generate_buffer.py).
BCQ.py: BCQ agent implementation. Called by main.py.
