from langchain_community.document_loaders import PyPDFLoader

# Load PDF
loader = PyPDFLoader("profile.pdf")  # make sure name matches your file
documents = loader.load()

# Print first page content
print(documents[0].page_content)