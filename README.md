# BCQ in TensorFlow

Authors' PyTorch BCQ implementation: https://github.com/sfujim/BCQ of BCQ paper https://arxiv.org/abs/1812.02900

* utils.py: Contains replay buffer.
* train_expert.py: Call this file to train a DDPG agent using DDPG.py.
* generate_buffer.py: Call this file to generate a replay buffer using an agent (for example one created using train_expert.py)
* main.py: Call this file to train a BCQ agent. Requires a replay buffer to be loaded (like one generated from generate_buffer.py).
* BCQ.py: BCQ agent implementation. Called by main.py.

Example usage from terminal:
``` 
#train expert
python3 train_expert.py --env_name LunarLanderContinuous-v2 --seed 31 --expl_noise=0.1 --actor_lr=0.0005 --critic_lr=0.0005 --actor_hs=64 --critic_hs=64 --batch_size=32 --discount=0.99 --tau=0.001
```
``` 
#generate buffer
python3 generate_buffer.py --env_name LunarLanderContinuous-v2 --seed 31 --actor_lr=0.0005 --critic_lr=0.0005 --actor_hs=64 --critic_hs=64 --batch_size=32 --discount=0.99 --tau=0.001 --noise1=0.1 --noise2=0.1
```
```
#train BCQ agent
python3 main.py --env_name LunarLanderContinuous-v2 --seed 31
```
