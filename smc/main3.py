from environment import Environment
from sp_baseline import SPBaseline
from utils.logger import Logger

if __name__ == '__main__':
    env = Environment()
    logger = Logger()
    model = SPBaseline(env, logger)

    model.run(max_trials=70)