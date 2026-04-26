
sys_prompt ='''You are an intelligent agent playing a game. 
               Your task is to open 5 boxes using 13 keys in fewest attempts. 
               You do not need special skills to play this game. This game can be played by an 8-12 year old child.'''

env_prompt ='''
            For each box there is a key that opens it, so the goal of the game is to find the right key for each box, using as few actions as possible.
            You have a demonstration video from a teacher telling you how to open all boxes. In the video, the teacher says:
            "I'm going to show you the right way to unlock the doors. To open the doors, you have to use a key that matches the color of the box. So, to open this red box, I'm going to use this red key. Great, now you can open all the doors!”

            Each key has an identifier (id) and a color. Each key also has either a number or a shape, but not both.

            Each box has a colour. It also has a shape, which is printed on at least one of its faces.
            Not all faces are visible to you initially, but the game allows you to pick up boxes and examine them to get more information.

            Here are the boxes, lined up in this order:
            The first box is red, has a moon shape.
            The second box is pink, has a cloud shape.
            The third box is white, has a diamond shape.
            The fourth box is purple, has a heart shape.
            The fifth box is blue, has a triangle shape.
            
            Here are the 13 keys (in no specific order):
            The red key is red and has the number 1.
            The pink key is pink and has the number 6.
            The grey2 key is grey and has the number 2.
            The cloud key is grey and has a cloud shape.
            The orange4 key is orange and has the number 4.
            The green3 key is green  and has the number 3.
            The blue key is blue  and has a star shape.
            The yellow5 key is yellow and has the number 5.
            The heart key is green and has a heart shape.
            The white key is white and has the number 7.
            The triangle key is yellow and has a triangle shape.
            The diamond key is orange and has a diamond shape.
            The purple key is purple and has an arrow shape.
            '''

action_prompt = [
    '''
    You can interact with this environment by taking two types of actions: 
    (1) Attempt Action: write Python code that will be used to generate opening attempts, or 
    (2) Observe Action: request more information about a given box.

    To Take the Observe Action, your output should be exactly 
    "PICK UP x", where x is the box id (do not use any other attributes of the box)
    To take an Attempt Action, you will need to write a Python function that specifies a hypotheses about which keys open which boxes.
    Your output should use the given starter code, and complete the function called predict according to its signature. Your output should contain only the Python program for predict, absolutely nothing else. It should NOT contain the Key or Box classes.
    ''',
    '''
    Here is history of actions taken and observed evidence. Please use them to make your decision:
    ''',
]

starter_code_prompt ='''
               Here is the starter code for attempt action

               def predict(key, box) -> bool:
                    # key is a Key object
                    # box is a Box object
                    # fill in your code

                class Key:
                    def __init__(self, id: str, color: str, number: Optional[int], shape: Optional[str]):
                        self.id = id
                        self.color = color
                        self.number = number
                        self.shape = shape

                class Box:
                    def __init__(self, id: str, color: str, shape: str, number: set, position: int):
                        self.id = id
                        self.color = color
                        self.shape = shape
                        self.position = position
                        self.number: set
                        # Current belief over number of shapes on this box.
                        # Unknown until PICK UP action is taken.
                        # After PICK UP: collapses to a singleton set containing the exact value.
                        # In predict(), handle both uncertain and certain cases.
                
                
                Here are the accurate data for all keys and boxes. The indexing of the arrays correspond to the line-up order of boxes and keys.

                key_data = {
                    "id": ["red", "pink", "grey2", "cloud", "orange4", "green3", "blue", "yellow5", "heart", "white", "triangle", "diamond", "purple"],
                    "color": ["red", "pink", "grey", "grey", "orange", "green", "blue", "yellow", "green", "white", "yellow", "orange", "purple"],
                    "number": [1, 6, 2, None, 4, 3, None, 5, None, 7, None, None, None],
                    "shape": [None, None, None, "cloud", None, None, "star", None, "heart", None, "triangle", "diamond", "arrow"]
                }

                box_data = {
                    "id": ["red", "pink", "white", "purple", "blue"],
                    "color": ["red", "pink", "white", "purple", "blue"],
                    "shape": ["moon", "cloud", "diamond", "heart", "triangle"], 
                    "number": [set(), set(), set(), set(), set()],
                    "position": [1, 2, 3, 4, 5],
                }
                '''
