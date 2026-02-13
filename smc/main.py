from smc import Engine
from environment import Environment
from generator import Generator


class Logger:
    def __init__(self, logging: bool = True):
        self.logging = True
    
    def log(self, log_str: str):
        if self.logging:
            print(log_str)

smc_config = {
    "num_particles": 10,
    "theta": 0.9,
    "ess_threshold": 0.5,
    "mode": "soc",
    "prior": "uniform",
}
max_trials = 5

if __name__ == '__main__':
    llm_generator = Generator()
    environment = Environment(include_inspect=False)
    logger = Logger(logging=True)

    smc_engine = Engine(smc_config, environment, llm_generator, logger)
    smc_engine.run(max_trials=max_trials)