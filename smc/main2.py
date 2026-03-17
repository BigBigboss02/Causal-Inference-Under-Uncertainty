from smc_sp import Engine
from environment import Environment
from utils.plotter import Plotter2
from llm.llm import LLM

class Logger:
    def __init__(self, logging: bool = True):
        self.logging = True
    
    def log(self, log_str: str):
        if self.logging:
            print(log_str)

smc_config = {
    "num_particles": 5,
    "init_theta": (0.5, 0.5),
    "ess_threshold": 0.5,
    'act_mode': 'sample'
}

llm = LLM()

max_trials = 70

if __name__ == '__main__':

    environment = Environment(include_inspect=False)
    logger = Logger(logging=True)

    smc_engine = Engine(smc_config, environment, llm, logger)
    history = smc_engine.run(max_trials=max_trials)

    plotter2 = Plotter2(history)
    plotter2.plot_boxes_opened_over_trials(show=True)
    plotter2.plot_hypothesis_probs_over_trials(show=True)
    plotter2.plot_theta_over_trials(show=True)