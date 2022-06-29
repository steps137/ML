import numpy as np
import matplotlib.pyplot as plt
import ray


ray.init(ignore_reinit_error=True)


import shutil
ENV_NAME = "BipedalWalker-v3"

CHECKPOINT_ROOT = "tmp/"+ENV_NAME
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

RAY_RESULTS = "tmp/ray_results/"
shutil.rmtree(RAY_RESULTS, ignore_errors=True, onerror=None)


import ray.rllib.agents.ppo as trainer

config = trainer.DEFAULT_CONFIG.copy()

config['num_workers'] = 1           # 8 parallel workers
config['num_sgd_iter'] = 50 
config['sgd_minibatch_size'] = 250
config['model']['fcnet_hiddens'] = [512, 512]

agent = trainer.PPOTrainer(config, env=ENV_NAME)


MAX_ITER     = 100000                 # max iterations
MAX_EPISODES = 1000                   # max episodes
file_name, episode, history = "", 0, []
#print(config)

best_mean = -100000
for it in range(1,MAX_ITER+1):
    res = agent.train()    
    
    episode += res['episodes_this_iter']
    
    mean = res['episode_reward_mean']
    history.append([episode, mean])
    if mean > best_mean:
        file_name = agent.save(CHECKPOINT_ROOT)
        best_mean = mean
    
    if it % 10 == 0:
        print(f"\r{it:3d} episode:{episode:5d}  reawrd: {mean:6.2f}  ({res['episode_reward_min']:6.2f}, {res['episode_reward_max']:6.2f}), best: {best_mean:.2f}  {file_name:30s}", end="")        
        
    if episode > MAX_EPISODES:
        print("\nfinish")
        break

history = np.array(history)        
plt.title(f"{ENV_NAME} best: {mean:.2f}", fontsize=18)
plt.plot(history[:,0], history[:,1])
plt.show()