import colorspacious as cs
# https://colorspacious.readthedocs.io/en/latest/tutorial.html#color-similarity

# Define box colors in RGB format
teal      =  [10, 82, 90] 
cream     =  [98, 98, 93]
pink      = [100,53,67]

green_key =  [60, 81, 73]  
blue_key  =  [31, 70, 97]
purple_key = [62, 56, 72]
yellow_key = [100,94, 81]
white_key  = [94, 96, 95]
grey_key   = [83, 83, 85]
pink_key   = [100, 69, 67]

print(f"The CAM02-UCS color difference")

print("teal box, and")
d = cs.deltaE(teal, green_key, input_space="sRGB255")
print(f"\tgreen keytag: {d:.2f}")
d = cs.deltaE(teal, blue_key, input_space="sRGB255")
print(f"\tblue keytag: {d:.2f}")
d = cs.deltaE(teal, purple_key, input_space="sRGB255")
print(f"\tpurple keytag: {d:.2f}")

print(f"cream box, and")
d = cs.deltaE(cream, yellow_key, input_space="sRGB255")
print(f"\tyellow keytag: {d:.2f}")
d = cs.deltaE(cream, white_key, input_space="sRGB255")
print(f"\twhite keytag: {d:.2f}")
d = cs.deltaE(cream, grey_key, input_space="sRGB255")
print(f"\twhite keytag: {d:.2f}")

print(f"cream box, and")
d = cs.deltaE(pink, pink_key, input_space="sRGB255")
print(f"\tpink keytag: {d:.2f}")