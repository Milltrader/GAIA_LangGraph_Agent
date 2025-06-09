from langchain_community.document_loaders import WikipediaLoader
page = WikipediaLoader(query="russia", load_max_docs=5, doc_content_chars_max=20000).load()[0].page_content
print(page[:50000 ])      