import ast
from environment import Key, Box

def check_valid_program(code: str) -> bool:

    # check syntax
    try:
        ast.parse(code)
    except SyntaxError:
        print(f'Invalid program generated:\n{code}')
        return False
    
    # check executable
    namespace = {}
    try:
        exec(code, namespace)
    except Exception:
        print(f'Invalid program generated:\n{code}')
        return False
    
    return True



def execute_hypothesis_code(code: str, key: Key, box: Box) -> bool:

    # assume functional code
    namespace = {}
    exec(code, namespace)
    return namespace['predict'](key, box)