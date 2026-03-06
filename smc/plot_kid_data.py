import json
from utils.plotter import KidDataPlotter


DATA_PATH = r"data\Dolly_KeyEviModel_7.3.24.json"

plot_config = {
    "min_attempts": 9,          # select IDs with > 10 attempts
    "strictly_more": True,       # True means > 10, False means >= 10
    "first_n_attempts": 10,      # plot only first 10 attempts
    "show_histogram": True,
    "show_lines": True,
}


def load_kid_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    kid_data = load_kid_data(DATA_PATH)
    plotter = KidDataPlotter(kid_data)

    selected_ids = plotter.get_ids_with_min_attempts(
        min_attempts=plot_config["min_attempts"],
        strictly_more=plot_config["strictly_more"],
    )

    print("Selected IDs:")
    print(selected_ids)

    if plot_config["show_histogram"]:
        plotter.plot_boxes_opened_histogram(show=True)

    if plot_config["show_lines"]:
        plotter.plot_first_n_attempts_for_selected_ids(
            selected_ids=selected_ids,
            first_n=plot_config["first_n_attempts"],
            show=True,
        )