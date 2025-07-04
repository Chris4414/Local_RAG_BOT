from haystack import Document
from sys import getsizeof
from pathlib import Path
import json

#Reads and saves json to a new Documents list
New_List = []

#Create a relative path for the jsonl file
base_file = Path(__file__).parent.parent
pdf_folder = base_file/"data"

#opens the jsonl file and saves documents to a list
with open(pdf_folder/"embedded_chunks.jsonl", "r") as f:
    New_List = [Document.from_dict(json.loads(line)) for line in f]

#prints chunk 27 just to verify that data is loadded into the jsonl
print(getsizeof(New_List))
print(New_List[27].content)
print(New_List[27].embedding)
