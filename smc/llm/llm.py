import os
from typing import List, Tuple

from llm.code import execute_hypothesis_code, check_valid_program
from llm.prompt import env_prompt, sys_prompt, generate_prompt, refine_prompt, hyp_prompt
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class LLM:
    def __init__(self):
        
        self.h_idx = 0

    def get_completion(self, sys_prompt: str, user_prompt: str, k: int = 1):

        response = client.responses.create(
            model='gpt-4',
            input=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        output_texts = list()
        for out in response.output:
            if out.type == 'message':
                for c in out.content:
                    if c.type == 'output_text':
                        output_texts.append(c.text)
        return output_texts[0]

    def generate(self, evidence: List) -> str:
        
        if len(evidence) == 0:
            user_prompt = env_prompt + hyp_prompt + generate_prompt
            # repeat until a valid hypothesis is generated
            while True:
                hypothesis = self.get_completion(sys_prompt, user_prompt)

                print(f'hypothesis generated: {hypothesis}')

                if check_valid_program(hypothesis):
                    self.h_idx += 1
                    return hypothesis, f'generated_{self.h_idx}'

    def refine(self, evidence: List, old_h: str) -> str:
        
        # construct prompt
        user_prompt = env_prompt + hyp_prompt + refine_prompt[0] + old_h + refine_prompt[1]
        for (key, box, outcome) in evidence:
            if outcome is True:
                user_prompt += f'Key {key.id} successfully opened box {box.id}\n'
            else:
                user_prompt += f'Key {key.id} failed to open box {box.id}\n'
        user_prompt += refine_prompt[2]

        # repeat until a valid hypothesis is generated
        while True:
            hypothesis = self.get_completion(sys_prompt, user_prompt)

            print(f'hypothesis refined: {hypothesis}')

            if check_valid_program(hypothesis):
                self.h_idx += 1
                return hypothesis, f'generated_{self.h_idx}'
            