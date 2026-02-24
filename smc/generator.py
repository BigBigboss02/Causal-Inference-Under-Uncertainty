from typing import Dict, List, Tuple, Any, Optional
from environment import Environment, Key, Box
import random

class Generator:

    def __init__(self, config: Dict, env: Environment):

        self.env: Environment = env
        self.omega = config['omega']
        self.prop_random = config['prop_random']
        
        self.hypotheses: Dict[str, Dict] = dict()

        self._build_proposal_dist()


    def _build_proposal_dist(self) -> None:

        self.distribution = list()
        keys, boxes = self.env.keys, self.env.boxes

        number_match = dict()
        for k in keys:
            if k.number is None:
                continue
            for b in boxes:
                if b.number is not None and b.number == k.number:
                    number_match[k.id] = b.id

        colour_match = dict()
        for k in keys:
            for b in boxes:
                if k.colour == b.colour:
                    colour_match[k.id] = b.id
        
        shape_match = dict()
        for k in keys:
            if k.shape is None:
                continue
            for b in boxes:
                if b.shape is not None and b.shape == k.shape:
                    shape_match[k.id] = b.id
        
        order_match = {"red": "red", "grey2": "pink", "green3": "white", "orange4": "purple", "yellow5": "blue",}

        # 14 hypothesis for similar colors
        fixed = {"purple": "purple", "pink": "pink", "red": "red"}

        sim_colour = list()
        sim_colour.append({**fixed, "heart": "blue", "white": "white"})        
        sim_colour.append({**fixed, "heart": "blue", "yellow5": "white"})    
        sim_colour.append({**fixed, "heart": "blue", "triangle": "white"})   
        sim_colour.append({**fixed, "heart": "blue", "grey2": "white"})       
        sim_colour.append({**fixed, "heart": "blue", "cloud": "white"})       
        sim_colour.append({**fixed, "blue": "blue", "yellow5": "white"})      
        sim_colour.append({**fixed, "blue": "blue", "triangle": "white"})     
        sim_colour.append({**fixed, "blue": "blue", "grey2": "white"})        
        sim_colour.append({**fixed, "blue": "blue", "cloud": "white"})        
        sim_colour.append({**fixed, "green3": "blue", "white": "white"})      
        sim_colour.append({**fixed, "green3": "blue", "yellow5": "white"})    
        sim_colour.append({**fixed, "green3": "blue", "triangle": "white"})   
        sim_colour.append({**fixed, "green3": "blue", "grey2": "white"})      
        sim_colour.append({**fixed, "green3": "blue", "cloud": "white"})  
        
        self.hypotheses["colour_match"] = colour_match
        self.hypotheses["shape_match"] = shape_match
        self.hypotheses["order_match"] = order_match
        self.hypotheses["number_match"] = number_match
        for i, h in enumerate(sim_colour):
            self.hypotheses[f"similar_colour_{i+1}"] = h

        # assign prior probability
        prior_colour = 5 * self.omega / 2
        prior_order = 5
        prior_shape = 2
        prior_number = 1
        prior_sim_colour = 5 * self.omega / 2 / 14

        prior_sum = prior_colour + prior_order + prior_shape + prior_number + prior_sim_colour * 14

        prob_colour = prior_colour / prior_sum * (1 - self.prop_random)
        prob_order = prior_order / prior_sum * (1 - self.prop_random)
        prob_shape = prior_shape / prior_sum * (1 - self.prop_random)
        prob_number = prior_number / prior_sum * (1 - self.prop_random)
        prob_sim_colour = prior_sim_colour / prior_sum * (1 - self.prop_random)
        
        self.distribution.append({ "name": "generator", "type": "generator", "prior": self.prop_random, "prob": self.prop_random })
        self.distribution.append({ "name": "colour_match", "type": "colour", "prior": prob_colour, "prob": prob_colour })
        self.distribution.append({ "name": "order_match", "type": "order", "prior": prob_order, "prob": prob_order })
        self.distribution.append({ "name": "shape_match", "type": "shape", "prior": prob_shape, "prob": prob_shape })
        self.distribution.append({ "name": "number_match", "type": "number", "prior": prob_number, "prob": prob_number })
        for i, h in enumerate(sim_colour):
            self.distribution.append({ "name": f"similar_colour_{i+1}", "type": "sim_colour", "prior": prob_sim_colour, "prob": prob_sim_colour })


    def sample(self) -> Tuple:
        weights = [h['prob'] for h in self.distribution]
        sampled_h = random.choices(self.distribution, weights=weights, k=1)[0]
        return sampled_h['name'], sampled_h['type'], sampled_h['prior']

    def sample_from_dist(self, n: int) -> List[Tuple]:
        weights = [h['prob'] for h in self.distribution]
        sampled_hs = random.choices(self.distribution, weights=weights, k=n)
        return [(h['name'], h['type'], h['prior']) for h in sampled_hs]
    
    def generate(self, evidence: List) -> Tuple[Dict, str]:

        def _sample_key_for_box(hypothesis: Dict, box: Box) -> None:
            """
            ASSUMPTION: probability among unused keys are uniform
            """
            unused_keys = [k for k in self.env.keys if k.id not in hypothesis]
            key = random.choice(unused_keys)
            hypothesis[key.id] = box.id

        # for opened boxes, key-box pairs are fixed in hypothesis
        hypothesis = dict()
        for key, box, outcome in evidence:
            if outcome is True:
                hypothesis[key.id] = box.id

        not_opened_boxes = [box for box in self.env.boxes if box.id not in hypothesis.values()]

        # complete hypothesis by sampling keys for unopened boxes
        for box in not_opened_boxes:
            _sample_key_for_box(hypothesis, box)

        return hypothesis, 'generator'
