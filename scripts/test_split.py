
import random, shutil
from pathlib import Path

val = Path("data/val")
test = Path("data/test")
test.mkdir(exist_ok=True)
classes = [d.name for d in val.iterdir() if d.is_dir()]
for c in classes:
    (test/c).mkdir(parents=True, exist_ok=True)
    imgs = [p for p in (val/c).glob("*") if p.is_file()]
    k = max(1, int(0.1*len(imgs)))
    move = random.sample(imgs, k)
    for p in move:
        shutil.move(str(p), str(test/c/p.name))
print("Created data/test with ~10% samples per class.")
