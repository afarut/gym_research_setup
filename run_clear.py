import shutil
import json

def clear():
    shutil.rmtree("./saved_models", ignore_errors=True)
    shutil.rmtree("./outputs", ignore_errors=True)

    with open("meta.json", "w", encoding="utf-8") as f:
        json.dump([], f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    clear()