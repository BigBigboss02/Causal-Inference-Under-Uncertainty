from smc import Engine
from environment import Environment
from generator import Generator


class Logger:
    def __init__(self, logging: bool = True):
        self.logging = True
    
    def log(self, log_str: str):
        if self.logging:
            print(log_str)

gen_config = {
    "omega": 2.0,
    "prop_random": 0.1
}

smc_config = {
    "num_particles": 30,
    "theta_distribution": (0.5, 0.5),
    "ess_threshold": 0.5,
    "k_rejuvenate": 30,
    "mode": "soc",
    "prior": "uniform",
}
max_trials = 500

if __name__ == '__main__':
    environment = Environment(include_inspect=False)
    generator = Generator(gen_config, environment)
    logger = Logger(logging=True)

    smc_engine = Engine(smc_config, environment, generator, logger)
    smc_engine.run(max_trials=max_trials)