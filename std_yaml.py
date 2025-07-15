# Read storage tech discord dictionary YAML file
import yaml

with open('assets/dictionary.yaml', 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
for entry in data:
    term = entry.get('term', '')
    tags = entry.get('tags', [])
    description = entry.get('description', '')
    print(f"term: {term}")
    print(f"tags: {tags}")
    print(f"description: {description}")
    print('-' * 40)

# write to excel file
import pandas as pd

df = pd.DataFrame()
# [0] = term, [1] = empty, [2] = tags + description
df['term'] = [entry.get('term', '') for entry in data]
df['empty'] = [''] * len(data)  # 添加空列
df['tags_description'] = [''] * len(data)  # 添加空列
for index, entry in enumerate(data):
    content = entry.get('description', '')
    if len(entry.get('tags', [])) > 0:
        content = 'tags: ' + ', '.join(entry.get('tags', [])) + '. - ' + content
    df.loc[index, 'tags_description'] = content
df.to_excel('assets/dictionary.xlsx', index=False)
