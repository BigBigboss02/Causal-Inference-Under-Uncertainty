default_hypothesis_codes = [
    """
        def evaluate(key, box): 
            return key.color == box.color
    """,
    """
        def evaluate(key, box): 
            return key.id == "red" and box.id =="red"
    """,
    """
        def evaluate(key, box): 
            return key.number is not None and key.number > 2
    """,
    """
        def evaluate(key, box):
            return False if key.shape is None else key.shape in box.shapes
    """
]


class Generator:

    def __init__(self):
        pass