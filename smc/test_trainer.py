from trainer import BoxTaskTrainer, SkillOnlyParams

D = [
    ("red",     "red",    1),
    ("pink",    "pink",   0),
    ("grey2",   "pink",   1),
    ("purple",  "purple", 0),
    ("green3",  "purple", 1),
    ("blue",    "blue",   0),
    ("yellow5", "blue",   1),
    ("white",   "white",  0),
    ("orange4", "white",  1),

    ("cloud",   "pink",   0),
    ("diamond", "white",  0),
    ("heart",   "purple", 0),
    ("triangle","blue",   0),

    ("grey2",   "pink",   1),
    ("green3",  "purple", 1),
    ("yellow5", "blue",   1),
    ("orange4", "white",  1),

    ("red",     "pink",   0),
    ("grey2",   "red",    0),
    ("red",     "red",    1),
]


trainer = BoxTaskTrainer(D, num_particles=20, seed=0)

alphas = [0.5, 1.0, 2.0, 5.0]
betas  = [0.5, 1.0, 2.0, 5.0]

best_params = None
best_ll = float("-inf")

print("Grid search over (alpha0, beta0) with 20 particles\n")

for a in alphas:
    for b in betas:
        params = SkillOnlyParams(alpha0=a, beta0=b)
        ll = trainer.log_likelihood(params)
        print(f"alpha0={a:>4}, beta0={b:>4} -> ll={ll:.6f}")
        if ll > best_ll:
            best_ll = ll
            best_params = (a, b)

print("\nBEST:")
print(f"(alpha0, beta0)=({best_params[0]}, {best_params[1]}), ll={best_ll:.6f}")