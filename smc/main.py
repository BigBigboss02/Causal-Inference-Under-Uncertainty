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
    "theta_distribution": (0.5, 0.5),
    "ess_threshold": 0.5,
    "mode": "soc",
    "prior": "uniform",
    "k_rejuvenate": 1, #fake
}
generator_config = {
    "omega": 1.0,          # fake
    "prop_random": 0.2     # fake
}
environment = Environment(include_inspect=False)

max_trials = 5

if __name__ == '__main__':
    # llm_generator = Generator()
    llm_generator = Generator(generator_config, environment)
    environment = Environment(include_inspect=False)
    logger = Logger(logging=True)

    smc_engine = Engine(smc_config, environment, llm_generator, logger)
    smc_engine.run(max_trials=max_trials)