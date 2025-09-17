import requests
import minsearch

def get_docs():
    '''
    Fetch documents from the provided URL and return as a list of dictionaries.
    '''
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    documents = []

    for course_dict in documents_raw:
        for doc in course_dict['documents']:
            doc['course'] = course_dict['course']
            documents.append(doc)

    return documents

def build_index(documents):
    '''
    Build a Minsearch index from the provided documents.
    '''

    index = minsearch.Index(
        text_fields=["question", "text", "section"],
        keyword_fields=["course"]
    )

    index.fit(documents)

    return index

def search(query):
    '''
    search function to get relevant documents from the index.
    '''
    boost = {'question': 3.0, 'section': 0.5}

    index = build_index(get_docs())

    results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=5
    )

    return results

def build_prompt(query):
    '''
    Build prompt for RAG based on the search results.
    Uses the search function to get relevant documents and constructs a prompt.
    '''

    prompt_template = """

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = ""
    
    for doc in search(query):
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()

    return prompt
