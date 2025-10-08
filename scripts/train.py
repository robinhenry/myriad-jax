# In a main experiment script (e.g., train.py)
from aion.configs.default import Config
from aion.platform.runner import train_and_evaluate

# 1. Create a default config
config = Config()

# # 2. Customize it for a specific experiment if needed
# config = config.replace(  # type: ignore
#     total_timesteps=100000,
#     agent=AgentConfig(name="pqn_agent", learning_rate=1e-3)
# )

# 3. Pass the single config object to your runner
train_and_evaluate(config)
