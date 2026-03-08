import ast

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

    # check dummy input
    func = namespace['try_open']
    try:
        result = func('dummy_key', 'dummy_box')
    except Exception:
        print(f'Invalid program generated:\n{code}')
        return False

    # return boolean return
    if not isinstance(result, bool):
        print(f'Invalid program generated:\n{code}')
        return False
    return True

def execute_hypothesis_code(code: str, key_id: str, box_id: str) -> bool:

    # assume functional code
    namespace = {}
    exec(code, namespace)
    return namespace['try_open'](key_id, box_id)