from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
import os
import re
import nltk
from nltk.corpus import stopwords
import morfeusz2
from elasticsearch import Elasticsearch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, TextStreamer, TextIteratorStreamer, AutoModelForSequenceClassification
import torch
from typing import List
from langchain_core.documents.base import Document
import spacy
from sentence_transformers import util
from tqdm import tqdm
import time
from threading import Thread
# from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
# from magic_pdf.data.dataset import PymuDocDataset
# from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
import shutil
from bs4 import BeautifulSoup
from io import BytesIO
from pdfminer.high_level import extract_text_to_fp
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import requests

def extract_pdf_lite(pdf_path):
    root = os.path.dirname(pdf_path)
    name = os.path.basename(pdf_path)

    # Extract pdf content to html
    output_buffer = BytesIO()
    with open(pdf_path, 'rb') as pdf_file:
        extract_text_to_fp(pdf_file, output_buffer, output_type='html')
    html_text = output_buffer.getvalue().decode('utf-8')

    # Convert HTML to txt 
    text_content = BeautifulSoup(html_text, "lxml").get_text()

    # Save the .txt file 
    text_path = os.path.join(root, f"{name}.txt")
    content_stripped = os.linesep.join([s for s in text_content.splitlines() if s])
    with open(text_path, "w", encoding="utf-8") as file:
        file.write(content_stripped)
    
    return text_path

# def extract_pdf(pdf_path):
#     root = os.path.dirname(pdf_path)
#     name = os.path.basename(pdf_path)
#     local_image_dir, local_md_dir = os.path.join(root, "images"), root
#     os.makedirs(local_image_dir, exist_ok=True)
#     image_save_dir = str(os.path.basename(local_image_dir))

#     # if File already exists, skip
#     txt_path = os.path.join(local_md_dir, f"{name}.txt")
#     if os.path.exists(str(os.path.join(local_md_dir, f"{name}.txt"))):
#         print(f"{txt_path} file already exists!")
#         return

#     image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

#     # Read the PDF content
#     reader = FileBasedDataReader("")
#     pdf_bytes = reader.read(str(os.path.join(root, name)))

#     # Create dataset instance
#     ds = PymuDocDataset(pdf_bytes, lang="pl")

#     # Inference
#     infer_result = ds.apply(doc_analyze, ocr=False, lang="pl")
#     pipe_result = infer_result.pipe_txt_mode(image_writer)

#     # Save extracted content
#     pipe_result.dump_md(md_writer, f"{name}.txt", image_save_dir)

#     # Remove images directory
#     shutil.rmtree(local_image_dir, ignore_errors=True)

#     return txt_path


'''
Suggested models for text generation:
- speakleash/Bielik-11B-v2.3-Instruct
- CYFRAGOVPL/Llama-PLLuM-8B-chat
- CYFRAGOVPL/PLLuM-12B-instruct
'''
class RAG:
    def __init__(self, es_index='kodeks_cywilny_256', gen_model='speakleash/Bielik-11B-v2.3-Instruct', llm_url=None):
        # Inicjalizacja zmiennych środowiskowych i załadowanie modeli
        self.morf = morfeusz2.Morfeusz()

        # Define the Elasticsearch index name
        self.index_name = es_index

        # Load SpaCy model for Polish NER
        self.nlp_pl = spacy.load("pl_core_news_sm")

        # Add llm url
        if llm_url is not None:
            self.llm_url = llm_url
            if 'LLM_URL' in os.environ:
                self.set_llm_service_url(os.getenv('LLM_URL'))

        # Set the ElasticSearch client
        if 'ES_URL' in os.environ and 'ES_KEY' in os.environ:
            self.set_database(os.getenv('ES_URL'), os.getenv('ES_KEY')) 

        # Load tokenizer and embedding model
        self.processing_tokenizer = AutoTokenizer.from_pretrained("Voicelab/sbert-base-cased-pl") # medicalai/ClinicalBERT
        self.embedding_model = AutoModel.from_pretrained("Voicelab/sbert-base-cased-pl") # medicalai/ClinicalBERT
        self.llm_loaded = False
        self.gen_model = gen_model
        model_name = "sdadas/polish-reranker-roberta-v2"
        self.rr_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.rr_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )

    # Set the ElasticSearch database credentials
    def set_database(self, es_url, es_key):
        self.es_client = Elasticsearch(
            es_url,
            api_key=es_key
        )

    # Change the database index
    def change_index(self, index_name):
        self.index_name = index_name

    # Get the current database index
    def get_index_name(self):
        return self.index_name

    def set_llm_service_url(self, llm_service_url):
        self.llm_url = llm_service_url

    # List recursivly all documents in the directory
    def list_files_recursive(self, directory):
        documents = []
        for root, _, files in os.walk(directory):
            for name in files:
                if name.lower().endswith('.txt'):
                    documents.append(str(os.path.join(root, name)))
                elif name.lower().endswith('.pdf'):
                    if os.path.exists(os.path.join(root, f"{name}.txt")):
                        #extracted_pdf_path = extract_pdf(str(os.path.join(root, name)))
                        extracted_pdf_path = extract_pdf_lite(str(os.path.join(root, name)))
                    else:
                        extracted_pdf_path = os.path.join(root, f"{name}.txt")
                    documents.append(str(extracted_pdf_path))
        return documents

    # Load the documents from directory tree
    def load_documents(self, docs_path):
        if os.path.isdir(docs_path):
            files = self.list_files_recursive(docs_path)
        elif docs_path.lower().endswith('.pdf'):
            extracted_pdf_path = extract_pdf_lite(docs_path)
            #extracted_pdf_path = extract_pdf(docs_path)
            files = [extracted_pdf_path]
        else:
            files = [docs_path]

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
        # stop_words = set(stopwords.words('polish'))
        # text_uni = ' '.join([word for word in text_uni.split() if word not in stop_words])
        
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
    def index_single_document(self, documents, index_name):
        self.es_client.index(
            index=index_name,
            document=documents
        )

    # Create specified index mapping in elasticsearch
    def create_index_mapping(self, index_name):
        mappings = {
            "properties": {
                "source": {
                    "type": "keyword"
                },
                "source_text": {
                    "type": "text"
                },
                "cleaned_text": {
                    "type": "text"
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768
                },
                "entities": {
                    "type": "nested",
                    "properties": {
                        "entity_name": {
                            "type": "keyword"
                        },
                        "entity_type": {
                            "type": "keyword"
                        }
                    }
                }
            }
        }

        self.es_client.indices.create(index=index_name, mappings=mappings)

    # Insert documents into the elasticsearch
    def insert_docs_dir(self, docs_root_dir, index_name, chunk_size=300, chunk_overlap=60):
        raw_documents = self.load_documents(docs_root_dir)
        # Split the documents
        split_raw_document_list = self.split_documents(raw_documents, chunk_size, chunk_overlap)

        # Process all documents
        embedded_docs_dict = self.process_documents(split_raw_document_list)
        
        # Check if index already exists if not, create one
        indexes = self.es_client.indices.get_alias(index="*")
        index_list = list(indexes.keys())

        if index_name not in index_list:
            self.create_index_mapping(index_name)

        # Insert all documents into ElasticSearch
        for embedded_doc in tqdm(embedded_docs_dict, desc="Inserting documents to ElasticSearch"):
            self.index_single_document(embedded_doc, index_name)

        time.sleep(1)
        print(f"Successfully loaded {len(raw_documents)}, splited into {len(split_raw_document_list)} and inserted to ElasticSearch under '{index_name}' index.")

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
        # stop_words = set(stopwords.words('polish'))
        # text = ' '.join([word for word in text.split() if word not in stop_words])
        
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

    # Retrieve documents
    def retrieve(self, query, ret_fun='similarity', retrieve_size=5, search_embed=True, query_cleaned=False):

        if ret_fun == 'similarity':
            search_fun = "cosineSimilarity(params.query_vector, 'embedding')"
        elif ret_fun == 'dotproduct':
            search_fun = "dotProduct(params.query_vector, 'embedding')"
        
        if query_cleaned == False:
            index_search = "source_text"
            query_text = query['source_text']
        else:
            index_search = "cleaned_text"
            query_text = query['cleaned_text']

        # Construct the search query using script_score for cosine similarity
        if search_embed:
            search_query = {
                "size": retrieve_size,  # Set the number of results you want to retrieve
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": search_fun,
                            "params": {"query_vector": query['embedding']}
                        }
                    }
                }
            }
        else:
            search_query = {
                "size": retrieve_size,  # Define the desired number of results
                "query": {
                    "match": {
                        index_search: query['source_text']  # Search within 'cleaned_text'
                    }
                }
            }

        # Execute the search
        retrieved_docs = self.es_client.search(index=self.index_name, body=search_query)
        retrieved_docs = retrieved_docs['hits']['hits']

        return retrieved_docs
    
    # Intitialate LLM
    def initiate_llm(self):
        self.chat_tokenizer = AutoTokenizer.from_pretrained(self.gen_model)
        self.chat_llm = AutoModelForCausalLM.from_pretrained(self.gen_model, torch_dtype=torch.bfloat16, device_map="auto")
        self.chat_streamer = TextIteratorStreamer(self.chat_tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.llm_loaded = True

    # Apply chat template to the query and documents
    def apply_template(self, query, documents, additional_instruct, use_rag):
        messages = []
        if use_rag:
            if self.gen_model=='speakleash/Bielik-11B-v2.3-Instruct':
                messages.append({"role": "system", "content": f"Odpowiedz na pytanie użytkownika posiłkując się dostarczonymi dla poszerzenia kontekstu dokumentami. {additional_instruct}"})
                #messages.append({"role": "system", "content": f"Na podstawie dostarczonych poniżej dokumentów odpowiedz na pytanie użytkownika które znajduję się na samym dole. Wnioskuj wyłącznie na podstawie dostarczonego kontekstu. Jeżeli nie jesteś w stanie odpowiedzieć na podstawie otrzymanych dokumentów uczciwie to powiedz. {additional_instruct}"})
                context_text = ""
                for id, doc in enumerate(documents):
                    context_text += f"# Dokument {id}: {doc['_source'].get('source_text')} "

                user_text=f"Odpowiedz na poniższe pytanie: {query['source_text']} \n\n ### Dokumenty dostarczone dla poszerzenia kontekstu: {context_text}"

                messages.append({"role": "user", "content": user_text})
            elif 'PLLuM' in self.gen_model:
                docs_text = ""
                for id, doc in enumerate(documents):
                    docs_text += f"Dokument {id}: {doc['_source'].get('source_text')}"
                
                user_msg = f'''
                Numerowana lista dokumentów jest poniżej:
                ---------------------
                <results>
                {docs_text}
                </results>
                ---------------------
                Odpowiedz na pytanie użytkownika wykorzystując tylko informacje znajdujące się w dokumentach, a nie wcześniejszą wiedzę.
                Udziel wysokiej jakości, poprawnej gramatycznie odpowiedzi w języku polskim. Odpowiedź powinna zawierać cytowania do dokumentów, z których pochodzą informacje. Zacytuj dokument za pomocą symbolu [nr_dokumentu] powołując się na fragment np. [0] dla fragmentu z dokumentu 0. Jeżeli w dokumentach nie ma informacji potrzebnych do odpowiedzi na pytanie, zamiast odpowiedzi zwróć tekst: "Nie udało mi się odnaleźć odpowiedzi na pytanie".
                {additional_instruct}

                Pytanie: {query['source_text']}
                '''
                messages.append({"role": "system", "content": f"Odpowiedz na pytanie użytkownika. {additional_instruct}"})
                messages.append({"role": "user", "content": user_msg})
            else:
                docs_text = ""
                for id, doc in enumerate(documents):
                    docs_text += f"Dokument {id}: {doc['_source'].get('source_text')}"

                user_msg = f'''
                {query['source_text']}

                ### Dokumenty dostarczone dla poszerzenia kontekstu:
                {docs_text}
                '''
                messages.append({"role": "system", "content": f"Odpowiedz na pytanie użytkownika. {additional_instruct}"})
                messages.append({"role": "user", "content": user_msg})
        else:
            messages.append({"role": "system", "content": f"Odpowiedz na pytanie użytkownika. {additional_instruct}"})
            messages.append({"role": "user", "content": f"Odpowiedz na poniższe pytanie. {additional_instruct}.\n Pytanie: {query['source_text']}"})

        return messages

    # Generate the answer
    def generate_answer(self, query, documents, additional_instruct="", max_new_tokens_v=1000, use_rag=True, verbose=0):

        # Define and load the models
        if not self.llm_loaded:
            self.initiate_llm()

        # Construct prompt template
        messages = self.apply_template(query, documents, additional_instruct, use_rag)
        if verbose >= 2:
            print(messages)

        # Generate answer
        inputs_tp = self.chat_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.chat_tokenizer(inputs_tp, return_tensors="pt", padding=True)

        attention_mask = inputs["attention_mask"]
    
        generation_kwargs = dict(inputs=inputs['input_ids'], attention_mask=attention_mask, pad_token_id=self.chat_tokenizer.eos_token_id, streamer=self.chat_streamer, max_new_tokens=max_new_tokens_v, do_sample=False)
        thread = Thread(target=self.chat_llm.generate, kwargs=generation_kwargs)
        thread.start()

        time.sleep(2)
        generated_text = ""
        for new_text in self.chat_streamer:
            generated_text += new_text
        
        if verbose >= 1:
            print(generated_text)

        return generated_text

    # Send the query to the LLM service
    def send_message(self, query, documents, additional_instruct="", use_rag=True):
        # Construct prompt template
        messages = self.apply_template(query, documents, additional_instruct, use_rag)

        # Construct data
        if 'localhost' in self.llm_url:
            data = {
                "system_message": messages[0]['content'],
                "user_message": messages[1]['content']
            }

            # Send the query to the LLM and acquire response
            try:
                response = requests.post(self.llm_url, json=data)
                response.raise_for_status()
                response = response.json()
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
            
            return response.get("model_response")
        else:
            data = {
                "messages": [
                    {"role": "system", "content": messages[0]['content']},
                    {"role": "user", "content": messages[1]['content']}
                ],
                "max_length": 1000,  # adjust as needed
                "temperature": 0.01
            }

            response = requests.put(
                self.llm_url,
                json=data,
                headers={
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                **self.auth_kwargs,
            )
            response.raise_for_status()
            response_json = response.json()

            return response_json['response']

    # Rerank retrieved documents based on amount of keywords from the query in the cleaned text
    def rerank_keywords(self, documents, query):
        query_keywords = query['cleaned_text'].split()

        for doc in documents:
            source_text = doc['_source']['cleaned_text']
            score_adjustment = sum(0.1 for keyword in query_keywords if keyword in source_text)
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
    
    # Rerank retrieved documents using LLM
    def rerank_llm(self, documents, query):
        texts = [f"{query['source_text']}</s></s>{doc['_source']['source_text']}" for doc in documents]
        tokens = self.rr_tokenizer(texts, padding="longest", truncation=True, return_tensors="pt")
        output = self.rr_model(**tokens)
        scores = output.logits.detach().float().numpy()
        scores = np.squeeze(scores).tolist()
        paired_docs_score = list(zip(documents, scores))
        sorted_paired_docs_score = sorted(paired_docs_score, key=lambda x: x[1], reverse=True)
        selected_documents = [document for document, score in sorted_paired_docs_score if score > 0]

        del output
        del scores
        del tokens
        del sorted_paired_docs_score
        return selected_documents
    
    # Main rerank documents function
    def rerank(self, documents, query, top_k=5, rr_entities=False, rr_keywords=False, rr_llm=True):

        reranked_documents = self.rerank_semantic(documents, query)

        if rr_entities:
            reranked_documents = self.rerank_entities(reranked_documents, query)
        
        if rr_keywords:
            reranked_documents = self.rerank_keywords(reranked_documents, query)
        
        if rr_llm:
            reranked_documents = self.rerank_llm(reranked_documents, query)

        return reranked_documents[:top_k]
    
    # Inference throught RAG
    def infer(self, query_text, additional_instruct="", max_new_tokens_v=1000, use_rag=True, top_k=10, retrieve_size=5, rr_entities=False, rr_keywords=False, rr_llm=True, ret_fun='similarity', search_embed=True, query_cleaned=False, verbose=0):

        # Preprocess query
        query = self.process_query(query_text)

        if use_rag:
            # Retrieve documents
            retrieved_docs = self.retrieve(query, ret_fun=ret_fun, search_embed=search_embed, query_cleaned=query_cleaned, retrieve_size=retrieve_size)
            # Re-rank documents
            reranked_docs = self.rerank(retrieved_docs, query, top_k, rr_entities, rr_keywords, rr_llm)
        else:
            reranked_docs = []

        # Generate answer
        if self.llm_url is None:
            answer = self.generate_answer(query, reranked_docs, additional_instruct=additional_instruct, max_new_tokens_v=max_new_tokens_v, use_rag=use_rag, verbose=verbose)
        else:
            answer = self.send_message(query, reranked_docs, additional_instruct=additional_instruct, use_rag=use_rag)

        return answer
    