from Space import SpaceEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback

class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        
    def _on_step(self):
        reward = self.locals['rewards'][0]
        self.logger.record('reward', reward)

        info = self.locals['infos'][0]

        ecc_avg = info['ecc_avg']
        sma_avg = info['sma_avg']
        fuel_avg = info['fuel_avg']

        self.logger.record('custom/ecc_avg', ecc_avg)
        self.logger.record('custom/sma_avg', sma_avg)
        self.logger.record('custom/fuel_avg', fuel_avg)

        return True

env = SpaceEnv(num_satellites=100, time_step=60, max_steps=2000)
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_tensorboard/", learning_rate =0.0001, ent_coef='auto_0.01')
callback = LoggingCallback()

model.learn(total_timesteps=100000, callback=callback, progress_bar=True)