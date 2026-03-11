class Box:
    def __init__(self, color, shape_count):
        self.color = color
        self.shape_count = shape_count

class Key:
    def __init__(self, identifier, color, shape=None, number=None):
        self.identifier = identifier
        self.color = color
        self.shape = shape
        self.number = number

# Define boxes: each box has a colour, a and between 1 to 5 shapes on its sides, which are numbered
boxes = {
    'red': Box('red', {'moon': 1}),
    'pink': Box('pink', {'cloud': 2}),
    'cream': Box('cream', {'diamond': 4}),
    'purple': Box('purple', {'heart': 3}),
    'teal': Box('teal', {'triangle': 5}),
}

# Define keys: each key has a coloured keytag; each keytags has sybmol - either a shape or a number
keys = {
    'red1': Key('red1', 'red', number=1),
    'pink6': Key('pink6', 'pink', number=6),
    'grey2': Key('grey2', 'grey', number=2),
    'greycloud': Key('greycloud', 'grey', shape='cloud'),
    'orange4': Key('orange4', 'orange', number=4),
    'green3': Key('green3', 'green', number=3),
    'bluestar': Key('bluestar', 'blue', shape='star'),
    'yellow5': Key('yellow5', 'yellow', number=5),
    'greenheart': Key('greenheart', 'green', shape='heart'),
    'white7': Key('white7', 'white', number=7),
    'triangleyellow': Key('triangleyellow', 'yellow', shape='triangle'),
    'diamondorange': Key('diamondorange', 'orange', shape='diamond'),
    'purplearrow': Key('purplearrow', 'purple', shape='arrow'),
}

# ground truth rules for opening boxes
def can_open_box(key_id, box_id):
    valid_combinations = {
        'red1': 'red',
        'grey2': 'pink',
        'orange4': 'cream',
        'green3': 'purple',
        'yellow5': 'teal',
    }
    
    return valid_combinations.get(key_id) == box_id
