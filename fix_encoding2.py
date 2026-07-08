with open("src/run_experiment2.py", "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace(
    'json.load(open(TEST_JSON)) + json.load(open(TRAIN_JSON))',
    'json.load(open(TEST_JSON, encoding="utf-8")) + json.load(open(TRAIN_JSON, encoding="utf-8"))'
)
with open("src/run_experiment2.py", "w", encoding="utf-8") as f:
    f.write(content)
print("수정 완료")