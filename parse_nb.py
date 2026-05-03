import json

with open("notebooks/GeoCalibAlign_Colab.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "train_lora_standard.py" in source or "lora_standard" in source or "finetune" in source:
            print(f"Cell {i+1} source:")
            print(source)
            if "outputs" in cell and len(cell["outputs"]) > 0:
                print(f"Cell {i+1} outputs:")
                for out in cell["outputs"]:
                    if "text" in out:
                        print("".join(out["text"])[-500:]) # print last 500 chars of text output
                    elif "data" in out and "text/plain" in out["data"]:
                        print("".join(out["data"]["text/plain"])[-500:])
            print("-" * 50)
