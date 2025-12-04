import re
import json

INPUT_PATH  = "InternVideo2.5_orig_pred.json"        # original internvideo json
OUTPUT_PATH = "InternVideo2.5_pred_clean.json"   # cleaned, valid JSON


def clean_file(in_path, out_path):
    # 1) Read raw text
    with open(in_path, "r", encoding="utf-8") as f:
        txt = f.read()

    # 2) Fix description lines: remove any inner " characters
    lines = txt.splitlines()
    fixed_lines = []

    for line in lines:
        if '"description"' in line:
            try:
                # find the colon and first/last double quotes on the line
                colon_idx = line.index(':')
                first_q = line.index('"', colon_idx)
                last_q = line.rfind('"')

                if last_q > first_q:
                    inner = line[first_q + 1:last_q]
                    # remove any stray " inside the description text
                    inner_fixed = inner.replace('"', "")
                    line = line[:first_q + 1] + inner_fixed + line[last_q:]
            except ValueError:
                pass

        fixed_lines.append(line)

    fixed_txt = "\n".join(fixed_lines)

    # 3) Remove trailing commas before } and ]
    #    e.g.  "Fallen Trees", ]   -> "Fallen Trees" ]
    #          "description": "...", } -> "description": "..."
    fixed_txt = re.sub(r",(\s*})", r"\1", fixed_txt)
    fixed_txt = re.sub(r",(\s*])", r"\1", fixed_txt)

    # 4) Verify that it is now valid JSON
    data = json.loads(fixed_txt)   # will raise if still invalid

    # 5) Dump as *proper* JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"Cleaned {len(data)} elements and saved to {out_path}")


if __name__ == "__main__":
    clean_file(INPUT_PATH, OUTPUT_PATH)
