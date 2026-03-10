from environment import Environment
from typing import List, Dict

from llm.llm import LLM
from llm.prompt import init_prompts, replace_prompts
from llm.code import check_valid_program

class Generator:

    def __init__(self, config: Dict, env: Environment, logger):
        
        self.env: Environment = env
        self.llm = LLM()

        self.logger = logger

        self.num_generated = 0
    
    def generate_initial(self):

        while True:
            code = self.llm.get_completion(
                sys_prompt=init_prompts['sys'],
                user_prompt=init_prompts['user']
            )
            if check_valid_program(code):
                break
        
        self.num_generated += 1
        h_name = f'generator_{self.num_generated}'

        return h_name, code
    

    def generate_replace(self, code: str, history: List):

        while True:
            code = self.llm.get_completion(
                sys_prompt=replace_prompts['sys'],
                user_prompt=replace_prompts['user1'] + code + replace_prompts['user2'] + history + replace_prompts['user3']
            )
            if check_valid_program(code):
                break
        
        self.num_generated += 1
        h_name = f'generator_{self.num_generated}'

        return h_name, code
        
