with open("src/data_loader.py", "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace('open(path, "r")', 'open(path, "r", encoding="utf-8")')
with open("src/data_loader.py", "w", encoding="utf-8") as f:
    f.write(content)
print("수정 완료")