fake_history = [
    {
        "t": 1,
        "opened": 0,
        "theta": 0.50,
        "probs": {
            "number_match": 0.30,
            "colour_match": 0.40,
            "similar_colour_1": 0.30
        }
    },
    {
        "t": 2,
        "opened": 1,
        "theta": 0.60,
        "probs": {
            "number_match": 0.55,
            "colour_match": 0.20,
            "similar_colour_1": 0.25
        }
    },
    {
        "t": 3,
        "opened": 2,
        "theta": 0.66,
        "probs": {
            "number_match": 0.80,
            "colour_match": 0.05,
            "similar_colour_1": 0.15
        }
    }
]

from plot import Plotter

plotter = Plotter(fake_history)
plotter.plot_boxes_opened_over_trials(show=True)
plotter.plot_hypothesis_probs_and_theta(show=True)