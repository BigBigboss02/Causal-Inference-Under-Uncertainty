import os
import re
from typing import List, Tuple

from llm.code import execute_hypothesis_code, check_valid_program
from llm.prompt import env_prompt, sys_prompt, generate_prompt, refine_prompt, hyp_prompt
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class LLM:
    def __init__(self, model: str = 'gpt-4o'):
        
        self.h_idx = 0
        self.model = model
        
        if 'gpt' in model:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        elif 'deepseek' in model:
            self.client = OpenAI(
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com"
            )
        else:
            #default to chat if unknown
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def _clean_response(self, text: str) -> str:
        #remove md code blocks if present, models sometimes respond
        code_match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        code_match = re.search(r'```\n(.*?)```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return text.strip()

    def get_completion(self, sys_prompt: str, user_prompt: str, k: int = 1):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return self._clean_response(response.choices[0].message.content)

    def generate(self, evidence: List) -> str:
        
        if len(evidence) == 0:
            user_prompt = env_prompt + hyp_prompt + generate_prompt
            # repeat until a valid hypothesis is generated
            while True:
                hypothesis = self.get_completion(sys_prompt, user_prompt)

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

            print('Refined')
            print(hypothesis)
            
            if check_valid_program(hypothesis):
                self.h_idx += 1
                return hypothesis, f'generated_{self.h_idx}'
            