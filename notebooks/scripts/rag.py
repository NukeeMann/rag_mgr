from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
import os
import re
import nltk
from nltk.corpus import stopwords
import morfeusz2
from elasticsearch import Elasticsearch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, TextStreamer, TextIteratorStreamer
import torch
from typing import List
from langchain_core.documents.base import Document
import spacy
from sentence_transformers import util
from tqdm import tqdm
import time
from threading import Thread


class RAG:
    def __init__(self, es_index='mgr_test_1'):
        # Inicjalizacja zmiennych środowiskowych i załadowanie modeli
        self.morf = morfeusz2.Morfeusz()

        # Define the Elasticsearch index name and client
        self.index_name = es_index
        self.es_client = Elasticsearch(
            os.getenv('ES_URL'),
            api_key=os.getenv('ES_KEY')
        )

        # Load SpaCy model for Polish NER
        self.nlp_pl = spacy.load("pl_core_news_sm")

        # Load tokenizer and embedding model
        self.processing_tokenizer = AutoTokenizer.from_pretrained("Voicelab/sbert-base-cased-pl")
        self.embedding_model = AutoModel.from_pretrained("Voicelab/sbert-base-cased-pl")
        self.llm_loaded = False

    # List recursivly all documents in the directory
    def list_files_recursive(self, directory):
        documents = []
        for root, _, files in os.walk(directory):
            for name in files:
                if '.txt' in name:
                    documents.append(str(os.path.join(root, name)))
        return documents

    # Load the documents from directory tree
    def load_documents(self, docs_path):
        files = self.list_files_recursive(docs_path)

        raw_documents = 0
        for file in files:
            loader = TextLoader(file)
            if not raw_documents:
                raw_documents = loader.load()
            else:
                raw_documents += loader.load()

        print(f"Loaded {len(raw_documents)} documents")
        return raw_documents

    # Split documents into smaller chunks
    def split_documents(self, documents, chunk_size=300, chunk_overlap=60):
        textSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
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
        
        print(f"Splited {len(documents)} documents into {len(split_docs)} chunks")
        return split_docs

    # Replace polish letters in string
    def replace_polish_letters(self, text): 
        polish_to_english = { 'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n', 'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z', 'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L', 'Ń': 'N', 'Ó': 'O', 'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z' } 
        for polish_char, english_char in polish_to_english.items(): 
            text = text.replace(polish_char, english_char) 
        return text

    # Named entity recognition in document
    def ner(self, text):
        ner_res = self.nlp_pl(text)
        found_entities = []
        for ent in ner_res.ents:
            found_entities.append({'entity_name': ent.text, 'entity_type': ent.label_})
        
        # Remove duplicates
        unique_entities = list({frozenset(entity.items()): entity for entity in found_entities}.values())
        return unique_entities

    # Clean text, remove polish words and extract entities
    def clean_doc(self, doc):
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
                analyses = self.morf.analyse(word)
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
        text_uni = self.replace_polish_letters(text_uni)
        
        # Remove unnecessary whitespace
        cleaned_text_uni = " ".join(text_uni.split())

        # Named Entity Recognition extraction in Polish
        entities = self.ner(cleaned_text_uni)
        
        # Convert page_content to lowercase
        cleaned_text_uni = cleaned_text_uni.lower()

        # Remove local directory path from source
        source_path = ''.join(doc.metadata['source'].split('/etipg/')[1:])
        
        return {'source_text': text, 'cleaned_text': cleaned_text_uni, 'entities': entities, 'source': source_path}

    # Tokenize and embedd the document content using LLM
    def embedd_doc(self, preprocessed_doc):
        inputs = self.processing_tokenizer(preprocessed_doc['cleaned_text'], return_tensors="pt", truncation=True, padding=True)
        outputs = self.embedding_model(**inputs)

        embedding = outputs.pooler_output.detach().numpy()[0]
        return {'source': preprocessed_doc['source'], 'text_embedded': embedding, 'cleaned_text': preprocessed_doc['cleaned_text'], 'source_text': preprocessed_doc['source_text'], 'entities': preprocessed_doc['entities']}

    # Process documents
    def process_documents(self, documents: List[Document]):
        # Preprocess each document
        preprocessed_docs = [self.clean_doc(doc) for doc in tqdm(documents, desc="Processing documents")]
        
        # Embedd the preprocessed documents
        embedded_docs = [self.embedd_doc(doc) for doc in tqdm(preprocessed_docs, desc="Embedding documents")]
        
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
    def index_single_document(self, documents):
        self.es_client.index(
            index=self.index_name,
            document=documents
        )

    def insert_docs_dir(self, docs_root_dir, chunk_size=300, chunk_overlap=60):
      # Load documents
      raw_documents = self.load_documents(docs_root_dir)

      # Split the documents
      split_raw_document_list = self.split_documents(raw_documents, chunk_size, chunk_overlap)

      # Process all documents
      embedded_docs_dict = self.process_documents(split_raw_document_list)
      
      # Insert all documents into ElasticSearch
      for embedded_doc in tqdm(embedded_docs_dict, desc="Inserting documents to ElasticSearch"):
          self.index_single_document(embedded_doc)

      time.sleep(1)
      print(f"Successfully loaded {len(raw_documents)}, splited into {len(split_raw_document_list)} and inserted to ElasticSearch under '{self.index_name}' index.")

    # Clear query text
    def process_query(self, query):
        # Remove new lines
        text = query.replace('\n', ' ')

        # Lemmatization
        lemmatized_words = []
        for word in text.split():
            try:
                # Morfeusz2 returns a list of tuples with morphological data
                analyses = self.morf.analyse(word)
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
        text = self.replace_polish_letters(text)

        # Look for entities
        query_entities = self.ner(text)

        # Convert to lower case
        text = text.lower()
        
        # Remove unnecessary whitespace
        cleaned_text = " ".join(text.split())

        # Embedd query
        query_embedding = self.embedd_query(cleaned_text)

        return {'source_text': query, 'cleaned_text': cleaned_text, 'embedding': query_embedding, 'entities': query_entities}

    # Embedd the query
    def embedd_query(self, text):
        inputs = self.processing_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        return outputs.pooler_output.detach().numpy()[0]

    def retrieve(self, query_text):
        query = self.process_query(query_text)

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
        retrieved_docs = self.es_client.search(index=self.index_name, body=search_query)
        retrieved_docs = retrieved_docs['hits']['hits']

        return retrieved_docs, query
    
    def initiate_llm(self, device, model_name="speakleash/Bielik-11B-v2.3-Instruct"):
        self.chat_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chat_llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
        #self.chat_streamer = TextStreamer(self.chat_tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.chat_streamer = TextIteratorStreamer(self.chat_tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.llm_loaded = True
  
    def generate_answer(self, query, documents, max_new_tokens_v=1000, additional_instruct=""):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Define and load the models
        if not self.llm_loaded:
            self.initiate_llm(device)

        # Construct prompt template
        messages = []
        messages.append({"role": "system", "content": "Jesteś asystentem AI specjalizującym się w analizie tekstu, który odpowiada na pytania korzystając z dostarczonych poniżej dokumentów. Twoje odpowiedzi powinny być krótkie, precyzyjne i w języku polskim. Jeżeli nie jesteś w stanie odpowiedzieć na pytanie na bazie dostarczonych dokumentów odpowiedz, że nie wiesz. Nie wymyślaj odpowiedzi jeżeli jej nie ma w tekście. Nie cytuj fragmentów z dostarczonych dokumentów. {additional_instruct}"})
        for id, doc in enumerate(documents):
            messages.append({"role": "system", "content": f"Dokument {id}: {doc['_source'].get('source_text')}"})
        messages.append({"role": "user", "content": query['source_text']})

        # Generate answer
        input_ids = self.chat_tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

        generation_kwargs = dict(inputs=input_ids, streamer=self.chat_streamer, max_new_tokens=max_new_tokens_v, do_sample=False)
        thread = Thread(target=self.chat_llm.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        for new_text in self.chat_streamer:
            generated_text += new_text
        
        return generated_text

    # Rerank retrieved documents based on amount of keywords from the query in the cleaned text
    def rerank_keywords(self, documents, query):
        query_keywords = query['cleaned_text'].split()

        for doc in documents:
            source_text = doc['_source']['cleaned_text']
            score_adjustment = sum(1 for keyword in query_keywords if keyword in source_text)
            doc['_score'] += score_adjustment  # Add to the current score

        # Sort documents based on the adjusted score
        return sorted(documents, key=lambda d: d['_score'], reverse=True)

    # Rerank retrieved documents based on entities occurances
    def rerank_entities(self, documents, query):
        query_entity_names = [item['entity_name'] for item in query['entities']]

        for doc in documents:
            matching_entities_score = 0
            doc_entities = doc['_source'].get('entities', [])
            
            # Calculate score based on matched entities
            for entity in doc_entities:
                if entity['entity_name'] in query_entity_names:
                    matching_entities_score += 0.2
            
            # Adjust the score by adding the matching entity score
            doc['_score'] += matching_entities_score
        
        # Sort documents based on the adjusted score
        return sorted(documents, key=lambda d: d['_score'], reverse=True)

    # Rerank retrieved documents using semantic similarity
    def rerank_semantic(self, documents, query):
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
    
    def rerank(self, documents, query):

        reranked_documents = self.rerank_semantic(documents, query)

        if query['entities']:
            reranked_documents = self.rerank_entities(reranked_documents, query)
        
        return reranked_documents
    
    def infer(self, query_text, additional_instruct=""):
        # Retrieve documents
        retrieved_docs, query = self.retrieve(query_text)

        # Re-rank documents
        reranked_docs = self.rerank(retrieved_docs, query)

        # Generate answer
        answer = self.generate_answer(query, reranked_docs, additional_instruct)

        return answer
        