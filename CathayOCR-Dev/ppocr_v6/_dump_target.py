import os
ROOT = r"E:\OCR综合\OCR待AI处理"
target = None
for dp, dn, fn in os.walk(ROOT):
    for f in fn:
        if "雍正" in f and "上谕" in f and f.endswith(".pdf"):
            target = os.path.join(dp, f)
            break
    if target:
        break
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_target_path.txt")
with open(out, "w", encoding="utf-8") as fh:
    fh.write(target or "")
print("TARGET_WRITTEN:", bool(target))
print("PATH:", target)
