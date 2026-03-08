baseline_prompts = {
    "system": '''You are an intelligent agent playing a game. 
                 Your task is to open 5 boxes using 13 keys in fewest attempts. 
                 You do not need special skills to play this game. This game can be played by an 8-12 year old child.''',
    "env": '''
            For each box there is a key that opens it, so the goal of the game is to find the right key for each box. 
            You have a demonstration video from a teacher telling you how to open all boxes. In the video, the teacher says:
            "I'm going to show you the right way to unlock the doors. To open the doors, you have to use a key that matches the color of the box. So, to open this red box, I’m going to use this red key. Great, now you can open all the doors!”

            Here are the boxes, lined up in this order:
            The red  box has 1 moon shape. 
            The pink box has 2 cloud shapes. Each cloud is numbered from 1 to 2.
            The cream (a color between yellow and white) box has 4 daimond shapes. Each daimond is numbered from 1 to 4.
            The purple box has 3 heart shapes. Each heart is numbered from 1 to 3.
            The teal (a color between green and blue) box has 5 triangle shapes. Each triangle is numbered from 1 to 5.


            Here are the 13 keys (in no specific order):
            The red1  key is red and has the number 1.
            The pink6 key is pink and has the number 6.
            The grey2 key is grey and has the number 2.
            The greycloud key is grey and has a cloud shape.
            The orange4 key is orange and has the number 4.
            The green3 key is green  and has the number 3.
            The bluestar key is blue  and has a star shape.
            The yellow5 key is yellow and has the number 5.
            The greenheart key is green and has a  heart shape.
            The white7 key is white and has the number 7.
            The triangleyellow key is yellow and has a triangle shape.
            The diamondorange key is orange  and has a diamond shape.
            The purplearrow key is purple and has an arrow shape.''',
    "generate": "",
    "try": ""
}

replace_prompts = []

user_prompt_base = 

user_prompt_initial = ''' 

Now is your turn to open the boxes. Which key will you try first?
Please respomd in the format "key, box" (e.g. "red1, red") and do not include any other text in the response.
'''
user_prompt_subsequent = ''' 

Which key will you try next? 
Please respomd in the format "key, box" (e.g. "red1, red") and do not include any other text in the response.'''


client = OpenAI(api_key=secret_keys.openAI_api_key)

completion =client.chat.completions.create(n=1, model="gpt-4",  # GPT-4o
                        messages=[ system_prompt,  {"role": "user", "content":  user_prompt_base + user_prompt_initial} ] )
        

# print(completion) # making sure there's just one response
msg = completion.choices[0].message
print(msg)
print(type(msg))

resp = msg.content

# keeping track of history
tried_key_box_combinations = ''
imaginary_keys = ''
imaginary_boxes = ''

open_boxes = []

# create a log of the key-box combinations that were tried and the outcome of each try
# open a log file
f = open("key_box_log.txt", "w")
f.write("key, box, outcome\n")

prompt_count = 0

while ( len(open_boxes) < 5 and prompt_count < 35 ):

    print("response: ")
    print (resp)
    prompt_count += 1

    # get the key-box combination from the response
    key_box_exists = True
    try :
        key, box = resp.split(", ")

        # is the suggested key in keys?
        if key not in kb.keys:
            print(f"You responded {key}, but there is no such key.")
            f.write(f"Non-existent key! {key}\n")
            key_box_exists = False
            imaginary_keys += f"The {key} key is not in the list of keys.\n"
        elif box not in kb.boxes:
            print(f"You responded {box}, but there is no such box.\n")
            f.write(f"Non-existent box! {box}\n")
            key_box_exists = False
            imaginary_boxes += f"The {box} box is not in the list of boxes.\n"

        if key_box_exists:
            # check if the key can open the box
            if kb.can_open_box(key, box):
                open_boxes.append(box)
                tried_key_box_combinations += f"The {key} opened the {box} box.\n"
                print(f"Opened the {box} box!")
                f.write(f"{key}, {box}, 1\n")
            else:
                tried_key_box_combinations += f"The {key} key did not open the {box} box.\n"
                print(f"The {key} key did not open the {box} box.")
                f.write(f"{key}, {box}, 0\n")
    except Exception as e:
        print("An error occurred: ", e)

    
    if (len(open_boxes) == 5): break

    s = ''
    if (len(tried_key_box_combinations) > 0):
        s = f'You have already tried the following keys-box combinations: {tried_key_box_combinations}\n'

    if (len(imaginary_boxes)) :
        s += imaginary_boxes

    if (len(imaginary_keys)) :
        s += imaginary_keys

    completion =client.chat.completions.create(n=1, model="gpt-4",  # GPT-4o
                messages=[ system_prompt,  {"role": "user", "content":  user_prompt_base + s + user_prompt_subsequent} ] )
        
    resp =  completion.choices[0].message.content

f.close()
print(tried_key_box_combinations)

    