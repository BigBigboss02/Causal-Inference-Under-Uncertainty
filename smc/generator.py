from typing import Dict, List, Tuple, Any, Optional
from environment import Environment, Key, Box
import random
from itertools import permutations

class Generator:

    def __init__(self, config: Dict, env: Environment):

        self.env: Environment = env
        self.omega = config['omega']
        self.prop_random = config['prop_random']
        
        self.hypotheses: Dict[str, Dict] = dict()
        self.num_generated = 0

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

        color_match = dict()
        for k in keys:
            for b in boxes:
                if k.color == b.color:
                    color_match[k.id] = b.id
        
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

        sim_color = list()
        sim_color.append({**fixed, "heart": "blue", "white": "white"})        
        sim_color.append({**fixed, "heart": "blue", "yellow5": "white"})    
        sim_color.append({**fixed, "heart": "blue", "triangle": "white"})   
        sim_color.append({**fixed, "heart": "blue", "grey2": "white"})       
        sim_color.append({**fixed, "heart": "blue", "cloud": "white"})       
        sim_color.append({**fixed, "blue": "blue", "yellow5": "white"})      
        sim_color.append({**fixed, "blue": "blue", "triangle": "white"})     
        sim_color.append({**fixed, "blue": "blue", "grey2": "white"})        
        sim_color.append({**fixed, "blue": "blue", "cloud": "white"})        
        sim_color.append({**fixed, "green3": "blue", "white": "white"})      
        sim_color.append({**fixed, "green3": "blue", "yellow5": "white"})    
        sim_color.append({**fixed, "green3": "blue", "triangle": "white"})   
        sim_color.append({**fixed, "green3": "blue", "grey2": "white"})      
        sim_color.append({**fixed, "green3": "blue", "cloud": "white"})  
        
        self.hypotheses["color_match"] = color_match
        self.hypotheses["shape_match"] = shape_match
        self.hypotheses["order_match"] = order_match
        self.hypotheses["number_match"] = number_match
        for i, h in enumerate(sim_color):
            self.hypotheses[f"similar_color_{i+1}"] = h

        # assign prior probability
        prior_color = 5 * self.omega / 2
        prior_order = 5
        prior_shape = 2
        prior_number = 1
        prior_sim_color = 5 * self.omega / 2 / 14

        prior_sum = prior_color + prior_order + prior_shape + prior_number + prior_sim_color * 14

        prob_color = prior_color / prior_sum * (1 - self.prop_random)
        prob_order = prior_order / prior_sum * (1 - self.prop_random)
        prob_shape = prior_shape / prior_sum * (1 - self.prop_random)
        prob_number = prior_number / prior_sum * (1 - self.prop_random)
        prob_sim_color = prior_sim_color / prior_sum * (1 - self.prop_random)
        
        self.distribution.append({ "name": "generator", "type": "generator", "prior": self.prop_random, "prob": self.prop_random })
        self.distribution.append({ "name": "color_match", "type": "color", "prior": prob_color, "prob": prob_color })
        self.distribution.append({ "name": "order_match", "type": "order", "prior": prob_order, "prob": prob_order })
        self.distribution.append({ "name": "shape_match", "type": "shape", "prior": prob_shape, "prob": prob_shape })
        self.distribution.append({ "name": "number_match", "type": "number", "prior": prob_number, "prob": prob_number })
        for i, h in enumerate(sim_color):
            self.distribution.append({ "name": f"similar_color_{i+1}", "type": "sim_color", "prior": prob_sim_color, "prob": prob_sim_color })


    def sample(self) -> Tuple:
        weights = [h['prob'] for h in self.distribution]
        sampled_h = random.choices(self.distribution, weights=weights, k=1)[0]
        return sampled_h['name'], sampled_h['type'], sampled_h['prior']

    def sample_from_dist(self, n: int) -> List[Tuple]:
        weights = [h['prob'] for h in self.distribution]
        sampled_hs = random.choices(self.distribution, weights=weights, k=n)
        return [(h['name'], h['type'], h['prior']) for h in sampled_hs]
    
    def prune_proposal_dist(self, key: Key, box: Box) -> None:
        """
        if key opens box, then remove all violating hypotheses from proposal
        """
        keep = list()
        for h in self.distribution:
            if h['type'] == 'generator':
                keep.append(True)
                continue
            if self.hypotheses.get(h['name'], None) is None:
                keep.append(True)
                continue
            violate = any((h_box_id == box.id and h_key_id != key.id) for h_key_id, h_box_id in self.hypotheses[h['name']].items())
            keep.append(not violate)
        
        self.distribution = [h for h, keep_h in zip(self.distribution, keep) if keep_h]
        
        # normalize prob
        total = sum(h['prob'] for h in self.distribution)
        if total > 0:
            for h in self.distribution:
                h['prob'] /= total


    def generate(self) -> Tuple[Dict, str]:

        def _sample_key_for_box(hypothesis: Dict, box: Box) -> None:
            """
            ASSUMPTION: probability among unused keys are uniform
            """
            unused_keys = [k for k in self.env.keys if k.id not in hypothesis]
            key = random.choice(unused_keys)
            hypothesis[key.id] = box.id

        # for opened boxes, key-box pairs are fixed in hypothesis
        hypothesis = dict()
        for key_id, box_id in self.env.success_pairs:
            hypothesis[key_id] = box_id

        not_opened_boxes = [box for box in self.env.boxes if box.id not in hypothesis.values()]

        # complete hypothesis by sampling keys for unopened boxes
        for box in not_opened_boxes:
            _sample_key_for_box(hypothesis, box)

        # check if generated hypothesis already exists
        for h_name in self.hypotheses:
            if self.hypotheses[h_name] == hypothesis:
                return hypothesis, h_name
        
        # if not exist
        self.num_generated += 1
        h_name = f'generator_{self.num_generated}'
        self.hypotheses[h_name] = hypothesis
        return hypothesis, h_name
    
    
    """
    NOT USED
    """
    def generate_non_duplicate(self, existing_h: List[Dict]) -> Tuple[Dict, str]:

        # for opened boxes, key-box pairs are fixed in hypothesis
        hypothesis = dict()
        for key_id, box_id in self.env.success_pairs:
            hypothesis[key_id] = box_id

        unopened = [box for box in self.env.boxes if box.id not in hypothesis.values()]
        unused_keys = [key for key in self.env.keys if key.id not in hypothesis.keys()]

        if len(unopened) == 0:
            return (None, None) if (hypothesis in existing_h) else (hypothesis, 'generator')
        
        # valid key assignments for unopened boxes that do not lead to duplicates
        candidates = list()
        for assigned_keys in permutations(unused_keys, len(unopened)):
            for key, box in zip(assigned_keys, unopened):
                hypothesis[key.id] = box.id
            if hypothesis not in existing_h:
                candidates.append(assigned_keys)
        
        if not candidates:
            return None, None
        
        assigned_keys = random.choice(candidates)
        for key, box in zip(assigned_keys, unopened):
            hypothesis[key.id] = box.id

        return hypothesis, 'generator'
        
