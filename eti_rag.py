from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
import os
import re
import nltk
from nltk.corpus import stopwords
import morfeusz2
from elasticsearch import Elasticsearch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, TextStreamer
import torch
from typing import List
from langchain_core.documents.base import Document
import spacy
from sentence_transformers import util
from tqdm import tqdm
import time


# Inicjalizacja zmiennych środowiskowych i załadowanie modeli
nltk.download('stopwords')
with open('stopwords-pl.txt', 'r', encoding='utf-8') as file:
    polish_stopwords = file.read().splitlines()
    
stopwords.words('polish').extend(polish_stopwords)
morf = morfeusz2.Morfeusz()

# Define the Elasticsearch index name
index_name = "mgr_test_1_embedded"

# Load SpaCy model for Polish NER
nlp_pl = spacy.load("pl_core_news_sm")

# Load tokenizer and embedding model
tokenizer = AutoTokenizer.from_pretrained("Voicelab/sbert-base-cased-pl")
embedding_model = AutoModel.from_pretrained("Voicelab/sbert-base-cased-pl")

# List recursivly all documents in the directory
def list_files_recursive(directory):
    documents = []
    for root, _, files in os.walk(directory):
        for name in files:
            if '.txt' in name:
                documents.append(str(os.path.join(root, name)))
    return documents

# Load the documents from directory tree
def load_documents(docs_path="/home/nukeemann/github/scrapper_mgr/etipg/eti.pg.edu.pl/"):
    files = list_files_recursive(docs_path)

    raw_documents = 0
    for file in files:
        loader = TextLoader(file)
        if not raw_documents:
            raw_documents = loader.load()
        else:
            raw_documents += loader.load()

    print(f"Załadowano {len(raw_documents)} dokumentów")
    return raw_documents

# Split documents into smaller chunks
def split_documents(documents):
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_docs = []
    for doc in documents:
        chunks = textSplitter.split_text(doc.page_content)
        for chunk in chunks:
            chunk_doc = Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,  # retain original metadata
                }
            )
            split_docs.append(chunk_doc)
    
    print(f"Podzielono {len(documents)} dokumentów na {len(split_docs)} chunków")
    return split_docs

# Replace polish letters in string
def replace_polish_letters(text): 
    polish_to_english = { 'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n', 'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z', 'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L', 'Ń': 'N', 'Ó': 'O', 'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z' } 
    for polish_char, english_char in polish_to_english.items(): 
        text = text.replace(polish_char, english_char) 
    return text

# Named entity recognition in document
def ner(text):
    ner_res = nlp_pl(text)
    found_entities = []
    for ent in ner_res.ents:
        found_entities.append({'entity_name': ent.text, 'entity_type': ent.label_})
    
    # Remove duplicates
    unique_entities = list({frozenset(entity.items()): entity for entity in found_entities}.values())
    return unique_entities

# Clean text, remove polish words and extract entities
def clean_doc(doc: Document):
    text = doc.page_content
    
    # Remove any leftover HTML tags
    clean_html = re.compile('<.*?>')
    text = re.sub(clean_html, '', text)
    
    # Remove new lines
    text = text.replace('\n', ' ')
    
    # Lemmatization
    lemmatized_words = []
    for word in text.split():
        try:
            analyses = morf.analyse(word)
            if analyses and len(analyses) == 1:
                lemma = analyses[0][2][1]
            else:
                lemma = word
        except Exception as e:
            print(f"An error occurred during analysis of word '{word}': {e}")
            lemma = word

        lemmatized_words.append(lemma)
    text_uni = ' '.join(lemmatized_words)
    
    # Remove Polish stopwords
    stop_words = set(stopwords.words('polish'))
    text_uni = ' '.join([word for word in text_uni.split() if word not in stop_words])
    
    # Replace polish letters
    text_uni = replace_polish_letters(text_uni)
    
    # Remove unnecessary whitespace
    cleaned_text_uni = " ".join(text_uni.split())

    # Named Entity Recognition extraction in Polish
    entities = ner(cleaned_text_uni)
    
    # Convert page_content to lowercase
    cleaned_text_uni = cleaned_text_uni.lower()

    # Remove local directory path from source
    source_path = ''.join(doc.metadata['source'].split('/etipg/')[1:])
    
    return {'source_text': text, 'cleaned_text': cleaned_text_uni, 'entities': entities, 'source': source_path}

# Tokenize and embedd the document content using LLM
def embedd_doc(preprocessed_doc):
    inputs = tokenizer(preprocessed_doc['cleaned_text'], return_tensors="pt", truncation=True, padding=True)
    outputs = embedding_model(**inputs)

    embedding = outputs.pooler_output.detach().numpy()[0]
    return {'source': preprocessed_doc['source'], 'text_embedded': embedding, 'cleaned_text': preprocessed_doc['cleaned_text'], 'source_text': preprocessed_doc['source_text'], 'entities': preprocessed_doc['entities']}

# Process documents
def process_documents(documents: List[Document]):
    # Preprocess each document
    preprocessed_docs = [clean_doc(doc) for doc in tqdm(documents, desc="Processing documents")]
    
    # Embedd the preprocessed documents
    embedded_docs = [embedd_doc(doc) for doc in tqdm(preprocessed_docs, desc="Embedding documents")]
    
    # Restructure the document list
    embedded_docs_dict = []
    for doc in embedded_docs:
        embedded_docs_dict.append({
            'source': doc['source'],
            'embedding': doc['text_embedded'],
            'source_text': doc['source_text'],
            'cleaned_text': doc['cleaned_text'],
            'entities': doc['entities']
        })
    
    return embedded_docs_dict

# Insert single document to the ElasticSearch databse
def index_single_document(index_name, document):
    es_client.index(
        index=index_name,
        document=document
    )
    #print(f"Document indexed: {response['_id']}")

# Clear query text
def process_query(query):
    # Remove new lines
    text = query.replace('\n', ' ')

    # Lemmatization
    lemmatized_words = []
    for word in text.split():
        try:
            # Morfeusz2 returns a list of tuples with morphological data
            analyses = morf.analyse(word)
            # Extract the lemma (base form) from the analysis results
            # Each 'analysis' tuple is structured as (position, (start_node, end_node, (base_form, tags, interp)))
            if analyses and len(analyses) == 1:
                lemma = analyses[0][2][1]
            else:
                lemma = word
        except Exception as e:
            # Handle any exceptions that might occur
            print(f"An error occurred during analysis of word '{word}': {e}")
            lemma = word
        # Append the lemma to the list
        lemmatized_words.append(lemma)
    text = ' '.join(lemmatized_words)
    
    # Remove Polish stopwords
    stop_words = set(stopwords.words('polish'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Replace polish letters
    text = replace_polish_letters(text)

    # Look for entities
    print(text)
    query_entities = ner(text)
    print(query_entities)

    # COnvert to lower case
    text = text.lower()
    
    # Remove unnecessary whitespace
    cleaned_text = " ".join(text.split())

    # Embedd query
    query_embedding = embedd_query(cleaned_text)

    return {'source_text': query, 'cleaned_text': cleaned_text, 'embedding': query_embedding, 'entities': query_entities}

# Embedd the query
def embedd_query(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.pooler_output.detach().numpy()[0]

def generate_answer(query, documents):
    # Define and load the models
    model_name = "speakleash/Bielik-11B-v2.3-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Construct prompt template
    messages = []
    messages.append({"role": "system", "content": "Jesteś asystentem AI, który odpowiada na pytania korzystając z dostarczonego kontekstu. Twoje odpowiedzi powinny być krótkie, precyzyjne i w języku polskim."})
    for doc in documents:
        messages.append({"role": "system", "content": f"Dokument ze strony {doc['_source'].get('source').replace('context.txt', '')}: {doc['_source'].get('source_text')}"})
    messages.append({"role": "user", "content": query['source_text']})

    # Generate answer
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
    llm.generate(input_ids, streamer=streamer, max_new_tokens=1000, do_sample=True)

# Rerank retrieved documents based on amount of keywords from the query in the cleaned text
def rerank_keywords(documents, query):
    query_keywords = query['cleaned_text'].split()

    for doc in documents:
        source_text = doc['_source']['cleaned_text']
        score_adjustment = sum(1 for keyword in query_keywords if keyword in source_text)
        doc['_score'] += score_adjustment  # Add to the current score

    # Sort documents based on the adjusted score
    return sorted(documents, key=lambda d: d['_score'], reverse=True)

# Rerank retrieved documents based on entities occurances
def rerank_entities(documents, query):
    query_entity_names = [item['entity_name'] for item in query['entities']]

    for doc in documents:
        matching_entities_score = 0
        doc_entities = doc['_source'].get('entities', [])
        
        # Calculate score based on matched entities
        for entity in doc_entities:
            if entity['entity_name'] in query_entity_names:
                matching_entities_score += 1
        
        # Adjust the score by adding the matching entity score
        doc['_score'] += matching_entities_score
    
    # Sort documents based on the adjusted score
    return sorted(documents, key=lambda d: d['_score'], reverse=True)

# Rerank retrieved documents using semantic similarity
def rerank_semantic(documents, query):
    # Convert embeddings to tensors
    document_embeddings_t = [torch.tensor(doc['_source']['embedding']) for doc in documents]
    query_embedding_t = torch.tensor(query['embedding']).unsqueeze(dim=0)

    # Compute similarity scores
    reranked_docs_semantic = []
    scores = util.semantic_search(query_embedding_t, document_embeddings_t, top_k=len(document_embeddings_t))

    for score in scores[0]:
        doc_index = score['corpus_id']
        doc = documents[doc_index]
        doc['_score'] = score['score']
        reranked_docs_semantic.append(doc)

    # Sort documents by the updated similarity score
    return sorted(reranked_docs_semantic, key=lambda x: -x['_score'])

# Load documents
raw_documents = load_documents("/home/nukeemann/github/scrapper_mgr/etipg/eti.pg.edu.pl/")
# Split the documents
split_raw_document_list = split_documents(raw_documents)

# Process all documents
embedded_docs_dict = process_documents(split_raw_document_list)

# Define low-level ElasticSearch client
es_client = Elasticsearch(
    cloud_id="",
    api_key=""
)

# Insert all documents into ElasticSearch
for embedded_doc in tqdm(embedded_docs_dict, desc="Inserting documents to ElasticSearch"):
    continue#index_single_document(index_name, embedded_doc)

time.sleep(1)

# Create and clean query
query_text = "habilitacja w dyscyplina AEEiTK prowadzona przez Miranda Rogoda-Zawiasa"
query = process_query(query_text)

# Construct the search query using script_score for cosine similarity
search_query = {
    "size": 5,  # Set the number of results you want to retrieve
    "query": {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding')",
                "params": {"query_vector": query['embedding']}
            }
        }
    }
}

# Execute the search
retrieved_docs = es_client.search(index=index_name, body=search_query)
retrieved_docs = retrieved_docs['hits']['hits']

# Rerank documents using keywords
reranked_docs_keyword = rerank_keywords(retrieved_docs, query)

# Rerank documents using NER
reranked_docs_entities = rerank_entities(retrieved_docs, query)

# Rerank documents using NER
reranked_docs_semantic = rerank_semantic(retrieved_docs, query)

# Process the response and print results
print("RESULTS OF INITIAL RANKING\n")
for i, hit in enumerate(retrieved_docs):
    print(f"Document {i}:")
    print(f"Score: {hit['_score']}")
    print(f"Source: {hit['_source'].get('source')}")
    print(f"Content: {hit['_source'].get('source_text')}")
    #print(f"Embeddings: {hit['_source'].get('embedding')}")
    print(f"Entities: {hit['_source'].get('entities')}")
    print("-" * 20)
print("="*40)

print("RESULTS OF KEYWORD RERANKING\n")
for i, hit in enumerate(reranked_docs_keyword):
    print(f"Document {i}:")
    print(f"Score: {hit['_score']}")
    print(f"Source: {hit['_source'].get('source')}")
    print(f"Content: {hit['_source'].get('source_text')}")
    #print(f"Embeddings: {hit['_source'].get('embedding')}")
    print(f"Entities: {hit['_source'].get('entities')}")
    print("-" * 20)
print("="*40)

print("RESULTS OF ENTITIES RERANKING")
for i, hit in enumerate(reranked_docs_entities):
    print(f"Document {i}:")
    print(f"Score: {hit['_score']}")
    print(f"Source: {hit['_source'].get('source')}")
    print(f"Content: {hit['_source'].get('source_text')}")
    #print(f"Embeddings: {hit['_source'].get('embedding')}")
    print(f"Entities: {hit['_source'].get('entities')}")
    print("-" * 20)
print("="*40)

print("RESULTS OF SEMANTIC LLM RERANKING\n")
for i, hit in enumerate(reranked_docs_semantic):
    print(f"Document {i}:")
    print(f"Score: {hit['_score']}")
    print(f"Source: {hit['_source'].get('source')}")
    print(f"Content: {hit['_source'].get('source_text')}")
    #print(f"Embeddings: {hit['_source'].get('embedding')}")
    print(f"Entities: {hit['_source'].get('entities')}")
    print("-" * 20)

generate_answer(query, reranked_docs_semantic)