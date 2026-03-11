rules = [
        # Rule 1: Match color
        {
            "rule_description": "If the key color matches the box color, it has a chance to open the box.",
            "rule_id": 1,
            "key_box_pairings": [
                ("red1", "red", 0.95),
                ("pink6", "pink", 0.95),
                ("white7", "cream", 0.95),
                ("purplearrow", "purple", 0.95),
                ("bluestar", "teal", 0.95)
            ]
        },
        
        # Rule 2: Match shape (if the key has a shape, it might correspond to a box with the same shape)
        {
            "rule_description": "If the key has a shape that matches a shape on the box, it has a chance to open the box.",
            "rule_id": 2,
            "key_box_pairings": [
                ("greycloud", "pink", 0.95),  # Cloud shape
                ("diamondorange", "cream", 0.95),  # Diamond shape
                ("greenheart", "purple", 0.95),  # Heart shape
                ("triangleyellow", "teal", 0.95)  # Triangle shape
            ]
        },

         # Rule 3: Match similar color 
        {
            "rule_description": "If the key color is similar to the box color, it has a chance to open the box.",
            "rule_id": 3,
            "key_box_pairings": [
                ("red1", "red", 0.95),
                ("pink6", "pink", 0.95),
                ("white7", "cream", 0.3),
                ("yellow5", "cream", 0.3),
                ("triangleyellow", "cream", 0.3),
                ("purplearrow", "purple", 0.95),
                ("bluestar", "teal", 0.3),
                ("green3", "teal", 0.3),
                ("greenheart", "teal", 0.3)
            ]
        }
        ]

for rule in rules:
    print(rule)
    print()

resp = [
    # Rule 1: Match color exactly
    {
        "rule_description": "If the key color exactly matches the box color, it has a high chance to open the box.",
        "rule_id": 1,
        "key_box_pairings": [
            ("red1", "red", 0.95),
            ("pink6", "pink", 0.95),
            ("white7", "cream", 0.95),
            ("purplearrow", "purple", 0.95),
            ("bluestar", "teal", 0.95)
        ]
    },
    
    # Rule 2: Match similar color or related attributes (shape/number)
    {
        "rule_description": "If the key has a color similar to the box or matches the shape/number attributes, it has a moderate chance to open the box.",
        "rule_id": 2,
        "key_box_pairings": [
            ("triangleyellow", "teal", 0.5),   # Yellow triangle key relates to triangle shapes in teal box
            ("yellow5", "cream", 0.5),         # Yellow key and cream box could be related
            ("diamondorange", "cream", 0.5),   # Orange key with diamond shape for diamond-shaped cream box
            ("greenheart", "purple", 0.5),     # Green key with heart shape for purple box with hearts
            ("greycloud", "pink", 0.5),        # Grey cloud key for pink box with cloud shapes
            ("orange4", "cream", 0.5)          # Orange key matching numbered box
        ]
    },
    
    # Rule 3: Numbers and shapes match
    {
        "rule_description": "If the key has the same number or shape as the box symbols, it has a moderate chance to open the box.",
        "rule_id": 3,
        "key_box_pairings": [
            ("red1", "red", 0.8),              # Number 1 matches the moon
            ("pink6", "pink", 0.8),            # Key numbered 6 could link to clouds numbered
            ("orange4", "cream", 0.8),         # Number 4 relates to diamonds on cream box
            ("green3", "purple", 0.8),         # Number 3 matches hearts on purple box
            ("yellow5", "teal", 0.8)           # Number 5 matches triangles on teal box
        ]
    }
]

resp1 = [
    # Rule 1: Exact color match with confirmed key
    {
        "rule_description": "If the key color exactly matches the box color and the key has been confirmed, it will open the box with certainty.",
        "rule_id": 1,
        "key_box_pairings": [
            ("red1", "red", 1.0),  # Confirmed correct key
            ("pink6", "pink", 0.95),
            ("white7", "cream", 0.95),
            ("purplearrow", "purple", 0.95),
            ("bluestar", "teal", 0.95)
        ]
    },
    
    # Rule 2: Match similar color or related attributes (shape/number)
    {
        "rule_description": "If the key has a color similar to the box or matches the shape/number attributes, it has a moderate chance to open the box.",
        "rule_id": 2,
        "key_box_pairings": [
            ("triangleyellow", "teal", 0.5),   # Yellow triangle key relates to triangle shapes in teal box
            ("yellow5", "cream", 0.5),         # Yellow key and cream box could be related
            ("diamondorange", "cream", 0.5),   # Orange key with diamond shape for diamond-shaped cream box
            ("greenheart", "purple", 0.5),     # Green key with heart shape for purple box with hearts
            ("greycloud", "pink", 0.5),        # Grey cloud key for pink box with cloud shapes
            ("orange4", "cream", 0.5)          # Orange key matching numbered box
        ]
    },
    
    # Rule 3: Numbers and shapes match
    {
        "rule_description": "If the key has the same number or shape as the box symbols, it has a moderate chance to open the box.",
        "rule_id": 3,
        "key_box_pairings": [
            ("pink6", "pink", 0.8),            # Key numbered 6 could link to clouds numbered
            ("orange4", "cream", 0.8),         # Number 4 relates to diamonds on cream box
            ("green3", "purple", 0.8),         # Number 3 matches hearts on purple box
            ("yellow5", "teal", 0.8)           # Number 5 matches triangles on teal box
        ]
    }
]


resp2 = [
    # Rule 1: Exact color match with confirmed key and exclusions
    {
        "rule_description": "If the key color exactly matches the box color and the key has been confirmed, it will open the box with certainty. Excluded keys based on failed attempts.",
        "rule_id": 1,
        "key_box_pairings": [
            ("red1", "red", 1.0),  # Confirmed correct key
            ("white7", "cream", 0.95),
            ("purplearrow", "purple", 0.95),
            ("bluestar", "teal", 0.95)
        ]
    },
    
    # Rule 2: Match similar color or related attributes (shape/number)
    {
        "rule_description": "If the key has a color similar to the box or matches the shape/number attributes, it has a moderate chance to open the box. The pink6 key is excluded from the pink box.",
        "rule_id": 2,
        "key_box_pairings": [
            ("triangleyellow", "teal", 0.5),   # Yellow triangle key relates to triangle shapes in teal box
            ("yellow5", "cream", 0.5),         # Yellow key and cream box could be related
            ("diamondorange", "cream", 0.5),   # Orange key with diamond shape for diamond-shaped cream box
            ("greenheart", "purple", 0.5),     # Green key with heart shape for purple box with hearts
            ("greycloud", "pink", 0.5),        # Grey cloud key for pink box with cloud shapes
            ("orange4", "cream", 0.5),         # Orange key matching numbered box
            ("pink6", "teal", 0.5)             # pink6 could be tested on other boxes like teal
        ]
    },
    
    # Rule 3: Numbers and shapes match
    {
        "rule_description": "If the key has the same number or shape as the box symbols, it has a moderate chance to open the box.",
        "rule_id": 3,
        "key_box_pairings": [
            ("pink6", "teal", 0.8),            # Key numbered 6 could relate to triangles on teal box
            ("orange4", "cream", 0.8),         # Number 4 relates to diamonds on cream box
            ("green3", "purple", 0.8),         # Number 3 matches hearts on purple box
            ("yellow5", "teal", 0.8)           # Number 5 matches triangles on teal box
        ]
    }
]

resp3 = [
    # Rule 1: Exact color match with confirmed keys and exclusions
    {
        "rule_description": "If the key color exactly matches the box color and the key has been confirmed, it will open the box with certainty. Excluded keys based on failed attempts.",
        "rule_id": 1,
        "key_box_pairings": [
            ("red1", "red", 1.0),    # Confirmed correct key for red box
            ("grey2", "pink", 1.0),  # Confirmed correct key for pink box
            ("white7", "cream", 0.95),
            ("purplearrow", "purple", 0.95),
            ("bluestar", "teal", 0.95)
        ]
    },
    
    # Rule 2: Match similar color or related attributes (shape/number)
    {
        "rule_description": "If the key has a color similar to the box or matches the shape/number attributes, it has a moderate chance to open the box. Excluded keys that have been confirmed or eliminated.",
        "rule_id": 2,
        "key_box_pairings": [
            ("triangleyellow", "teal", 0.5),   # Yellow triangle key relates to triangle shapes in teal box
            ("yellow5", "cream", 0.5),         # Yellow key and cream box could be related
            ("diamondorange", "cream", 0.5),   # Orange key with diamond shape for diamond-shaped cream box
            ("greenheart", "purple", 0.5),     # Green key with heart shape for purple box with hearts
            ("greycloud", "pink", 0.0),        # Excluded because grey2 opened the pink box
            ("orange4", "cream", 0.5)          # Orange key matching numbered box
        ]
    },
    
    # Rule 3: Numbers and shapes match
    {
        "rule_description": "If the key has the same number or shape as the box symbols, it has a moderate chance to open the box.",
        "rule_id": 3,
        "key_box_pairings": [
            ("orange4", "cream", 0.8),         # Number 4 relates to diamonds on cream box
            ("green3", "purple", 0.8),         # Number 3 matches hearts on purple box
            ("yellow5", "teal", 0.8),          # Number 5 matches triangles on teal box
            ("pink6", "teal", 0.8)             # pink6 could relate to triangles in teal box
        ]
    }
]
