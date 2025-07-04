from haystack import Document
from sys import getsizeof
import json

#Reads and saves json to a new Documents list
New_List = []
with open("embedded_chunks.jsonl", "r") as f:
    New_List = [Document.from_dict(json.loads(line)) for line in f]

print(getsizeof(New_List))
print(New_List[27].content)
print(New_List[27].embedding)
