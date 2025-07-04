from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import SentenceTransformersTextEmbedder
#from haystack import Document
from pathlib import Path
from haystack.components.preprocessors import DocumentSplitter
import json

# For PDFs
#pdf_converter = PyPDFToDocument()
#pdf_docs = pdf_converter.run(paths=["path/to/file1.pdf", "path/to/file2.pdf"])["documents"]

# For text files
#txt_converter = TextFileToDocument()
#txt_docs = txt_converter.run(paths=["path/to/file1.txt"])["documents"]


# Define the folder containing your PDFs
base_file = Path(__file__).parent.parent
pdf_folder = base_file/"PDF's"
#Path(r"C:\Users\syb3d\OneDrive\Python\3.11\Local_RAG_BOT\PDF's")

# Get all PDF files in the folder
pdf_files = list(pdf_folder.glob("*.pdf"))
print(pdf_files)

# Initialize the PDF converter 
pdf_converter = PyPDFToDocument()

# Convert PDFs to Haystack Document objects (Converts to Haystack Documents)
documents = pdf_converter.run(pdf_files)["documents"]

#Initialize the splitter function. Documents will be chunked in 400 word chunks with 50 overlapping words between chunks.
splitter = DocumentSplitter(split_by="word", split_length=400, split_overlap=50)

#Chunk all documents and store them into split_docs [Document Name, List of Document Chunks[] .content gives a str]
split_docs = splitter.run(documents=documents)

# Embed documents: Embed means to create a vector that represents each split doc
embedder = SentenceTransformersTextEmbedder(model="intfloat/e5-small-v2")
embedder.warm_up()

# Haystack Document.embedding is built on for the vector to be saved to
for key, value_list in split_docs.items():
    for item in value_list:
        var = embedder.run(f"passage: {item.content}") #This embedder returns a dictionary but we only want the attached list of floats
        item.embedding = var["embedding"] 

#Create a full list of embedded documents. Previous list is good for organizing but document store just wants a list of Hastack Documents
Embedded_List = []

for key, value_list in split_docs.items():
    Embedded_List.extend(value_list)

#Writes full embedded list to a New jsonl file
with open("embedded_chunks.jsonl", "a") as f:
    for item in Embedded_List:
        f.write(json.dumps(item.to_dict()) + "\n")

#Reads and saves json to a new Documents list
#New_List = []
#with open("embedded_chunks.jsonl", "r") as f:
    #New_List = [Document.from_dict(json.loads(line)) for line in f]
