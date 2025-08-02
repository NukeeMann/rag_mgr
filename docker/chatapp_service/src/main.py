import os
import time
import re

import streamlit as st
import torch
from streamlit_extras.bottom_container import bottom

from src.rag_service.rag import RAG

torch.classes.__path__ = []


def save_uploaded_file(uploaded_file):
    # Specify a directory to save the files
    save_dir = 'session_files'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the file to the specified directory
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def minimize_spacing():
    st.markdown(
        """
        <style>
        /* Remove margin between elements */
        .block-container div.stAlert {
            margin-bottom: 0;
        }
        /* Remove padding in the container */
        .css-1kyxreq {  /* Adjust this selector if Streamlit updates class names */
            padding: 0;
        }
        div[data-testid="InputInstructions"] > span:nth-child(1) {
            visibility: hidden;
        }
        /* Hide the Streamlit header and menu */
        header {visibility: hidden;}
        /* Optionally, hide the footer */
        .streamlit-footer {display: none;}
        /* Hide your specific div class, replace class name with the one you identified */
        .st-emotion-cache-uf99v8 {display: none;}
        div[data-testid="stVerticalBlock"]{
            gap: 0.5rem !important;
        }

        button[kind="secondary"]  {
            padding-top: 5px !important;
            padding-bottom: 5px !important;
            min-height: 1rem !important;
            max-height: 1.5rem !important;
            width: 12.5rem !important;
        }

        button[kind="secondary"]  p{
            font-size: 15px !important;
        }

        div[data-testid="stBottomBlockContainer"] {
            padding: 0rem 0rem 3.5rem !important;
        }

        div[data-testid="stMainBlockContainer"] {
            padding: 1rem 1rem 0rem 1rem !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0rem .5rem .5rem .5rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_app():
    with st.container(height=35, border=False):
        st.write('*Bielik v2.2 Instruct - RAG*')

    # Initialize chat history
    with st.container(border=False):
        if 'messages' not in st.session_state:
            st.session_state.messages = [{'role': 'assistant', 'content': "Let's start chatting! 👇"}]

        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.text(message['content'])

    # Initialize visibility state
    if 'show_form' not in st.session_state:
        st.session_state.show_form = False
        st.session_state.creds = False
        st.session_state.settings = False

    with bottom():
        # Accept user input
        s_settings = st.button('Settings')
        prompt = st.chat_input('What is up?')

        if s_settings:
            st.session_state.settings = not st.session_state.settings

        # Create setting container
        if st.session_state.settings:
            with st.container(border=True):
                st.write('#### Settings')
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("&nbsp;", unsafe_allow_html=True)
                    st.session_state.s_use_rag = st.checkbox('Use RAG', value=st.session_state.s_use_rag)
                    st.session_state.s_rr_entities = st.checkbox(
                        'Rerank - Entities', value=st.session_state.s_rr_entities
                    )

                with col2:
                    st.session_state.s_retrieve_size = st.number_input(
                        'Retrieve size',
                        label_visibility='collapsed',
                        placeholder='Retrieve (1-10)',
                        min_value=1,
                        max_value=10,
                        step=1,
                        value=st.session_state.s_retrieve_size,
                    )
                    st.session_state.s_rr_llm = st.checkbox('Rerank - LLM', value=st.session_state.s_rr_llm)

                with col3:
                    st.markdown("&nbsp;", unsafe_allow_html=True)
                    s_creds = st.button('Services Creds')
                    s_upload = st.button('Upload File')

                col4, col5 = st.columns([2, 1])  # Create columns with a 2:1 width ratio
                with col4:
                    st.session_state.s_additional_instructions = st.text_input(
                        'Additional Instructions:', value=str(st.session_state.s_additional_instructions)
                    )
                with col5:
                    index_list = st.session_state.rag_system.get_index_list()
                    st.session_state.s_index_name = st.selectbox(
                        'Index name',
                        index_list,
                    )

            # Button to reveal form
            if s_upload:
                st.session_state.show_form = True

            if s_creds:
                st.session_state.creds = True

        # Conditionally display the form
        if st.session_state.show_form:
            with st.form('file_upload_form'):
                uploaded_file = st.file_uploader('Choose a file', type=['txt', 'csv', 'xlsx', 'pdf'])

                # Additional parameters
                s_index_name_new = st.text_input("Enter index name", placeholder='np. my-index', help='Musi mieć od 3 do 25 znaków, zawierać tylko małe litery (a-z) i łączniki (-).')
                s_chunks_size = st.number_input('Enter chunk size:', min_value=0, max_value=1536, value=256)
                s_chunks_overlap = st.number_input('Enter chunk overlap:', min_value=0, max_value=1536, value=32)

                # Submit and Close buttons
                col1_upload, _, _, _, _, col2_upload = st.columns(6)
                with col1_upload:
                    submit_button = st.form_submit_button(label='Submit')
                with col2_upload:
                    close_button_upload = st.form_submit_button(label='Close')

            # Handle form submission
            if submit_button:
                if uploaded_file is not None:
                    if s_index_name_new:
                        if re.match(r'^[a-z-]{3,25}$', s_index_name_new):
                            st.write(f'Filename: {uploaded_file.name}')
                            st.write('Index name:', s_index_name_new)
                            st.write('Chunks size:', s_chunks_size)
                            st.write('Chunks overlap:', s_chunks_overlap)

                            # Add your file processing logic
                            st.session_state.show_form = False
                            with st.spinner('Processig the file...', show_time=True):
                                file_path = save_uploaded_file(uploaded_file)
                                st.session_state.rag_system.insert_docs_dir(
                                    file_path, s_index_name_new, s_chunks_size, s_chunks_overlap
                                )
                                time.sleep(5)
                            st.success(f"Done! Document has been inserted into '{s_index_name_new}' database")
                            st.button('Close')
                        else:
                            st.warning('Nieprawidłowa nazwa indeksu. Musi mieć od 3 do 25 znaków i zawierać tylko małe litery (a-z) oraz łączniki (-).')
                    else:
                        st.warning('Please specify the index name.')
                else:
                    st.warning('Please select a file.')

            if close_button_upload:
                st.session_state.show_form = False
                st.rerun()

        # Conditionally display the form
        if st.session_state.creds:
            with st.form('db_llm_service_creds'):
                # Pinecone database credentials
                s_api_key = st.text_input('Enter Pinecone API KEY')
                s_llm_service_url = st.text_input('Enter LLM Service URL')

                # Update and Close buttons
                col1_update, _, _, _, _, col2_update = st.columns(6)
                with col1_update:
                    update_button = st.form_submit_button(label='Update')
                with col2_update:
                    close_button_update = st.form_submit_button(label='Close')

            # Handle form submission
            if update_button:
                if s_api_key:
                    if len(s_api_key) > 5:
                        st.session_state.rag_system.set_database(s_api_key)
                        st.success('Done! The database and LLM service have been set')
                    else:
                        st.warning('Please provide ElasticSearch URL and KEY')

                if s_llm_service_url:
                    st.session_state.rag_system.set_llm_service_url(s_llm_service_url)
                    st.success('Done! LLM service have been set')

                st.button('Close')

            # Close form button
            if close_button_update:
                st.session_state.creds = False
                st.rerun()

    # If user input is provided
    prompt_container = st.container()
    with prompt_container:
        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            # Update databse index
            st.session_state.rag_system.change_index(st.session_state.s_index_name)

            st.session_state.add_instruct = st.session_state.s_additional_instructions

            # Generate response
            llm_response = st.session_state.rag_system.infer(
                query_text=prompt,
                additional_instruct=st.session_state.s_additional_instructions,
                use_rag=st.session_state.s_use_rag,
                retrieve_size=st.session_state.s_retrieve_size,
                rr_entities=st.session_state.s_rr_entities,
                rr_llm=st.session_state.s_rr_llm,
            )

            # Add LLM response to chat history
            st.session_state.messages.append({'role': 'assistant', 'content': llm_response})

            # Rerun session to display the messages
            st.rerun()


if __name__ == '__main__':
    # Initialize variables
    if 'initialized' not in st.session_state:
        s_index_name = 'kodeks-cywilny-256'
        llm_api_url = 'https://153.19.239.239/api/llm/prompt/chat'  # "http://localhost:8080/api/generate"
        st.session_state.rag_system = rag_system = RAG(es_index=s_index_name, llm_url=llm_api_url)
        st.session_state.s_index_name = 'kodeks-cywilny-256'
        st.session_state.s_additional_instructions = ''
        st.session_state.s_use_rag = True
        st.session_state.s_rr_entities = True
        st.session_state.s_retrieve_size = 5
        st.session_state.s_rr_llm = False
        st.session_state.initialized = True

    # Apply the CSS
    minimize_spacing()

    # Build app
    build_app()
