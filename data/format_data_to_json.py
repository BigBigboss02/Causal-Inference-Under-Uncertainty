import pandas as pd
import json
from collections import defaultdict
from typing import Dict, List, Tuple


def load_longform_to_dict(
    excel_path: str,
    sheet_name: str = "Long Form"
) -> Dict[str, List[Tuple[str, str, int, Dict[str, int]]]]:

    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    data = defaultdict(list)

    for _, row in df.iterrows():

        pid = str(row["ID"]).strip()

        key = str(row["Key"]).strip().lower()
        box = str(row["Box"]).strip().lower()

        # Worked column = outcome
        outcome = int(row["Worked"])

        match_flags = {
            "color": int(row["ColorMatch"]),
            "number": int(row["NumMatch"]),
            "shape": int(row["ShapeMatch"]),
        }

        data[pid].append((key, box, outcome, match_flags))

    return dict(data)


def save_dict_to_json(data: Dict, filepath: str):

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print("Saved JSON to:", filepath)
    print("Participants:", len(data))


def convert_longform_excel_to_json(
    excel_path: str,
    output_json_path: str,
    sheet_name: str = "Long Form"
):

    data = load_longform_to_dict(excel_path, sheet_name)

    save_dict_to_json(data, output_json_path)

    return data


if __name__ == "__main__":

    excel_file = r"data\Dolly_KeyEviModel_7.3.24.xlsx"
    output_file = r"data\Dolly_KeyEviModel_7.3.24.json"

    convert_longform_excel_to_json(excel_file, output_file)