default_hypothesis_id = [
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10
]

default_hypothesis_soc = [
  {
    "red": "red",
    "heart": "pink",
    "triangle": "cream",
    "blue": "purple",
    "yellow5": "teal"
  },
  {
    "cloud": "red",
    "pink": "pink",
    "diamond": "cream",
    "green3": "purple",
    "white": "teal"
  },
  {
    "orange4": "red",
    "triangle": "pink",
    "red": "cream",
    "heart": "purple",
    "blue": "teal"
  },
  {
    "grey2": "red",
    "cloud": "pink",
    "yellow5": "cream",
    "purple": "purple",
    "triangle": "teal"
  },
  {
    "white": "red",
    "pink": "pink",
    "heart": "cream",
    "diamond": "purple",
    "green3": "teal"
  },
  {
    "blue": "red",
    "orange4": "pink",
    "triangle": "cream",
    "cloud": "purple",
    "yellow5": "teal"
  },
  {
    "purple": "red",
    "heart": "pink",
    "diamond": "cream",
    "red": "purple",
    "green3": "teal"
  },
  {
    "triangle": "red",
    "white": "pink",
    "cloud": "cream",
    "orange4": "purple",
    "blue": "teal"
  },
  {
    "yellow5": "red",
    "pink": "pink",
    "heart": "cream",
    "green3": "purple",
    "triangle": "teal"
  },
  {
    "cloud": "red",
    "diamond": "pink",
    "white": "cream",
    "blue": "purple",
    "orange4": "teal"
  }
]


default_hypothesis_code = [
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

