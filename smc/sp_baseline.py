from environment import Environment
from llm.llm import LLM
from llm.prompt import baseline_prompts
from utils.code import check_valid_program, execute_hypothesis_code

class SPBaseline:

    def __init__(self, env: Environment, logger):

        self.env = env
        self.llm = LLM()
        self.logger = logger

        self.hypothesis: str = None
        self.evidence = list()

    def _generate_hyp(self):
        
        def _build_prompt():
            
            evidence_prompt = ""
            for (box, key, outcome) in self.evidence:
                if outcome is True:
                    evidence_prompt += f"{key.id} successfully opens {box.id}\n"
                else:
                    evidence_prompt += f"{key.id} fails to open {box.id}\n"
            return baseline_prompts['env'] + evidence_prompt + baseline_prompts['generate']
        
        gen_prompt = _build_prompt()
        
        # repeat until a valid program is generated
        while True:
            temp_h = self.llm.get_completion(baseline_prompts['system'], gen_prompt)
            if check_valid_program(temp_h):
                self.hypothesis = temp_h
                break
        
    def _select_action(self):

        

        key, box = 'dummy', 'dummy'
        return (key, box)
    
    def run(self, max_trials: int) -> bool:
        
        self.trial_count = 0

        while not self.env.is_solved() and self.trial_count < max_trials:

            self.logger.log(f"TRIAL {self.trial_count}")

            self._generate_hyp()

            self.logger.log(f"{self.hypothesis}")

            (key, box) = self._select_action()
            outcome = self.env.test_action(key, box)

            self.evidence.append((key, box, outcome))

            self.logger.log(f"Action chosen: ({key.id}, {box.id})")
            self.logger.log(f"Outcome: {outcome}")

            self.trial_count += 1