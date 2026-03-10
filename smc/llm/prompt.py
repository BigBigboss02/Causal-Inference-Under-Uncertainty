
sys_prompt ='''You are an intelligent agent playing a game. 
               Your task is to open 5 boxes using 13 keys in fewest attempts. 
               You do not need special skills to play this game. This game can be played by an 8-12 year old child.'''

env_prompt ='''
            For each box there is a key that opens it, so the goal of the game is to find the right key for each box. 
            You have a demonstration video from a teacher telling you how to open all boxes. In the video, the teacher says:
            "I'm going to show you the right way to unlock the doors. To open the doors, you have to use a key that matches the color of the box. So, to open this red box, I'm going to use this red key. Great, now you can open all the doors!”

            Here are the boxes, lined up in this order:
            The red box has 1 moon shape. 
            The pink box has 2 cloud shapes. Each cloud is numbered from 1 to 2.
            The white box has 4 diamond shapes. Each diamond is numbered from 1 to 4.
            The purple box has 3 heart shapes. Each heart is numbered from 1 to 3.
            The blue box has 5 triangle shapes. Each triangle is numbered from 1 to 5.

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
            
            When I say "the [X] key" or "the [X] box", X is called the id of the key/box. 
            '''

hyp_prompt ='''A hypothesis is a valid Python program that can be executed to predict the outcome of a given key and box.
               The Python program should have the following signature:

               def try_open(key: Key, box: Box) -> bool:
                    # fill in your code

                The Key and Box objects are defined as follow. Note

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
                
                where the default value for optional parameters is None.
                The number field in Box represents the number of shapes on the box.'''

generate_prompt = '''Now, it is your turn to generate a hypothesis.
                     Your hypothesis should be a Python program that contains exactly the try_open function, including the provided signature.

                     Your output should contain only the Python program, absolutely nothing else.
                     Your output should NOT contain the Key or Box classes.
                    '''

refine_prompt = [
    '''Now, your task is to improve and refine an existing hypothesis that performs poorly on existing evidence.
       This is the hypothesis:''',

    '''Here are the evidence from previous attempts:
    ''',

    """Generate a new hypothesis.
       Your hypothesis should be a Python program that contains exactly the try_open function, including the provided signature.

       Your output should contain only the Python program, absolutely nothing else.
       Your output should NOT contain the Key or Box classes.
    """
]
