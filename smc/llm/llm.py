import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class LLM:
    def __init__(self):
        pass

    def get_completion(self, sys_prompt: str, user_prompt: str, k: int = 1):

        response = client.response.create(
            model='gpt-4',
            input=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            n=k
        )

        output_texts = list()
        for out in response.output:
            if out.type == 'message':
                for c in out.content:
                    if c.type == 'output_text':
                        output_texts.append(c.text)
        return output_texts[0]

