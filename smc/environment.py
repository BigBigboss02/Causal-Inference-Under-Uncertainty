from typing import Optional, List, Tuple
import numpy as np
import random

"""
configuration of keys and boxes as used in the experiment
"""

key_data = {
    "id": ["red", "pink", "grey2", "cloud", "orange4", "green3", "blue", "yellow5", "heart", "white", "triangle", "diamond", "purple"],
    "color": ["red", "pink", "grey", "grey", "orange", "green", "blue", "yellow", "green", "white", "yellow", "orange", "purple"],
    "number": [1, 6, 2, None, 4, 3, None, 5, None, 7, None, None, None],
    "shape": [None, None, None, "cloud", None, None, "star", None, "heart", None, "triangle", "diamond", "arrow"]
}

box_data = {
    "id": ["red", "pink", "cream", "purple", "teal"],
    "color": ["red", "pink", "cream", "purple", "teal"],
    "shape": ["moon", "cloud", "diamond", "heart", "triangle"], 
    "count": [1, 2, 4, 3, 5],
    "position": [1, 2, 3, 4, 5],
}

key_box_mapping = {
    ("red", "red"), ("grey2", "pink"), ("orange4", "white"), ("yellow5", "blue"), ("green3", "purple")
}


class Key:
    def __init__(self, id: str, color: str, number: Optional[int], shape: Optional[str]):
        self.id = id
        self.color = color
        self.number = number
        self.shape = shape

class Box:
    def __init__(self, id: str, color: str, number: int, shape: str):
        self.id = id
        self.color = color
        self.number = number
        self.shape = shape


class Environment:
    def __init__(self, include_inspect: bool = False):

        self.include_inspect = include_inspect
        self.keys, self.boxes = self._create_keys_and_boxes()
        self.actions = self._get_all_possible_actions()

        self.id_to_key, self.id_to_box = dict(), dict()
        for key in self.keys:
            self.id_to_key[key.id] = key
        for box in self.boxes:
            self.id_to_box[box.id] = box

        self.success_pairs = set()

    def _create_keys_and_boxes(self):
        keys, boxes = list(), list()

        for i in range(len(key_data["id"])):
            keys.append(Key(
                id=key_data["id"][i],
                color=key_data["color"][i],
                number=key_data["number"][i],
                shape=key_data["shape"][i]
            ))
        for i in range(len(box_data["id"])):
            boxes.append(Box(
                id=box_data["id"][i],
                color=box_data["color"][i],
                number=box_data["count"][i],
                shape=box_data["shape"][i]
            ))

        return keys, boxes

    def _get_all_possible_actions(self):
        actions = [(key, box) for key in self.keys for box in self.boxes]
        if self.include_inspect:
            actions.extend([('inspect', box) for box in self.boxes])
        return actions

    def is_solved(self) -> bool:
        return (len(self.success_pairs) >= 5)

    def get_random_action(self) -> Tuple[Key, Box]:
        return random.choice(self.actions)

    def test_action(self, key: Key, box: Box) -> bool:
        if (key.id, box.id) in key_box_mapping:
            if (key.id, box.id) not in self.success_pairs:
                self.success_pairs.add((key.id, box.id))
            return True
        else:
            return False