import os
import re
from openai import OpenAI
from dotenv import load_dotenv
import keys_boxes as kb

# python.terminal.useEnvFile
load_dotenv()
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')


if deepseek_api_key:
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    model_name = "deepseek-chat"


system_prompt = {"role": "system", 
                 "content": '''You are an intelligent agent playing a game. 
Your task is to open 5 boxes using 13 keys in fewest attempts. 
You do not need special skills to play this game. This game can be played by an 8-12 year old child.'''
}

user_prompt_base = '''
For each box there is a key that opens it, so the goal of the game is to find the right key for each box. 
You have a demonstration video from a teacher telling you how to open all boxes. In the video, the teacher says:
"I’m going to show you the right way to unlock the doors. To open the doors, you have to use a key that matches the color of the box. So, to open this red box, I’m going to use this red key. Great, now you can open all the doors!”

Here are the boxes, lined up in this order:
The red box has 1 moon shape. 
The pink box has 2 cloud shapes. Each cloud is numbered from 1 to 2.
The cream (a color between yellow and white) box has 4 daimond shapes. Each daimond is numbered from 1 to 4.
The purple box has 3 heart shapes. Each heart is numbered from 1 to 3.
The teal (a color between green and blue) box has 5 triangle shapes. Each triangle is numbered from 1 to 5.

Here are the 13 keys (in no specific order):
The red1 key is red and has the number 1.
The pink6 key is pink and has the number 6.
The grey2 key is grey and has the number 2.
The greycloud key is grey and has a cloud shape.
The orange4 key is orange and has the number 4.
The green3 key is green and has the number 3.
The bluestar key is blue and has a star shape.
The yellow5 key is yellow and has the number 5.
The greenheart key is green and has a heart shape.
The white7 key is white and has the number 7.
The triangleyellow key is yellow and has a triangle shape.
The diamondorange key is orange and has a diamond shape.
The purplearrow key is purple and has an arrow shape.
'''

user_prompt_instructions = '''
We will play in turns. In each turn, you must FIRST explain your reasoning based on the teacher's advice and your past observations, and then provide your action.
Always use the following exact format:

Thought: [Your reasoning about what key to try next]
Action: key, box

(e.g., Action: red1, red)
'''


history = "Past Turns and Observations:\n"
open_boxes = []

with open("key_box_log.txt", "w") as f:
    f.write("key, box, outcome\n")

    prompt_count = 0
    
    while len(open_boxes) < 5 and prompt_count < 35:
        prompt_count += 1
        
        # Build the full prompt for this turn
        current_prompt = user_prompt_base + "\n" + history + "\n" + user_prompt_instructions
        
        completion = client.chat.completions.create(
            n=1, 
            model=model_name,
            messages=[system_prompt, {"role": "user", "content": current_prompt}]
        )
        
        resp = completion.choices[0].message.content.strip()
        print(f"\n--- Turn {prompt_count} ---")
        print(resp)
        
        # Parse the Action using regex
        action_match = re.search(r'Action:\s*([a-zA-Z0-9_]+),\s*([a-zA-Z0-9_]+)', resp, re.IGNORECASE)
        
        observation = ""
        if action_match:
            key = action_match.group(1).strip()
            box = action_match.group(2).strip()
            
            key_box_exists = True
            if key not in kb.keys:
                observation = f"Observation: Invalid Action. The {key} key is not in the list of keys."
                key_box_exists = False
            elif box not in kb.boxes:
                observation = f"Observation: Invalid Action. The {box} box is not in the list of boxes."
                key_box_exists = False
            elif box in open_boxes:
                observation = f"Observation: The {box} box is already open. Choose a different box."
                key_box_exists = False
                
            if key_box_exists:
                if kb.can_open_box(key, box):
                    open_boxes.append(box)
                    observation = f"Observation: Success! The {key} opened the {box} box."
                    f.write(f"{key}, {box}, 1\n")
                else:
                    observation = f"Observation: Failure. The {key} key did not open the {box} box."
                    f.write(f"{key}, {box}, 0\n")
        else:
            observation = "Observation: Formatting error. I could not parse your Action. Make sure to use the exact format 'Action: key, box'."
            
        print(observation)
        
        #keep history
        history += f"\nTurn {prompt_count}:\n{resp}\n{observation}\n"

print("\nGame Over!")
print(f"Total attempts: {prompt_count}")
print(f"Boxes opened: {len(open_boxes)}/5")
