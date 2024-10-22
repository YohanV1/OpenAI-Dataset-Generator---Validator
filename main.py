import streamlit as st
import json
import openai
import os
from json_validator import validate
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(layout="wide", page_title='OpenAI Dataset Generator & Validator')

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def to_jsonl(system_message, input_list, output_list):
    jsonl_str = ''
    for inp, out in zip(input_list, output_list):
        json_obj = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": inp},
                {"role": "assistant", "content": out}
            ]
        }
        jsonl_str += json.dumps(json_obj) + '\n'
    return jsonl_str


def save_jsonl(jsonl_str, filename):
    with open(filename, 'w') as f:
        f.write(jsonl_str)
    return filename

def upload_to_openai(filename):
    with open(filename, "rb") as f:
        file = openai.File.create(file=f, purpose="fine-tune")
    return file.id

st.subheader("Step 1: Create a fine-tuning dataset")
st.markdown("First, add system message that sets the context for the model.")
system_message = st.text_area("System Message")

if 'num_rows' not in st.session_state:
    st.session_state.num_rows = 10

with st.expander("Provide at least 10 training examples", expanded=False):
    input_list = []
    output_list = []
    for i in range(st.session_state.num_rows):
        col1, col2 = st.columns(2)
        with col1:
            input_text = st.text_area(f"User Input {i + 1}")
        with col2:
            output_text = st.text_area(f"AI Output {i + 1}")
        input_list.append(input_text)
        output_list.append(output_text)

    if st.button('Add More'):
        st.session_state.num_rows += 1
        st.rerun()

if st.button('Generate JSONL File'):
    jsonl_str = to_jsonl(system_message, input_list, output_list)
    st.session_state.jsonl_str = jsonl_str
    st.session_state.filename = 'generated_fine_tune_data.jsonl'
    if system_message and len(input_list) == len(output_list) and all(input_list) and all(output_list):
        save_jsonl(jsonl_str, st.session_state.filename)
        st.write(f"Saved as {st.session_state.filename}")
        jsonl_bytes = st.session_state.jsonl_str.encode('utf-8')
        st.download_button(
            label="Download JSONL File",
            data=jsonl_bytes,
            file_name=st.session_state.filename,
            mime="text/jsonl",
        )

def verify_data_format(dataset):
    validate(dataset)

st.subheader("Step 3: Verify Data")
if st.button('Verify Data') and st.session_state.jsonl_str:
    verify_data_format(st.session_state.jsonl_str)

st.subheader("Step 3: Upload to OpenAI")
if st.button('Upload to OpenAI') and st.session_state.jsonl_str and st.session_state.filename:
    file_id = upload_to_openai(st.session_state.filename)
    st.session_state.file_id = file_id
    st.write(f"Uploaded to OpenAI with file ID: {file_id}")

st.subheader("Step 4: Create Fine-Tuning Job")
if st.button('Create Fine-Tuning Job') and st.session_state.file_id:
    try:
        fine_tuning_job = openai.FineTuningJob.create(
            training_file=st.session_state.file_id,
            model="gpt-3.5-turbo"
        )
        st.session_state.fine_tuning_job_id = fine_tuning_job['id']
        st.write(f"Fine-tuning job created with ID: {fine_tuning_job['id']}")
    except openai.error.InvalidRequestError as e:
        st.write(f"Error: {e}")
        st.write("File is still being processed and is not ready for fine-tuning. Please try again later.")

st.subheader("Step 5: Retrieve Fine-Tuned Model")
if st.button('Retrieve Fine-Tuned Model') and st.session_state.fine_tuning_job_id:
    retrieved_job = openai.FineTuningJob.retrieve(st.session_state.fine_tuning_job_id)
    fine_tuned_model = retrieved_job.get("fine_tuned_model", None)

    st.markdown(f"**Fine-Tuned Model:** `{fine_tuned_model}`")
    st.markdown(f"**Status** {retrieved_job.get('status', 'N/A')}")
    st.markdown(f"**Object:** {retrieved_job.get('object', 'N/A')}")
    st.markdown(f"**ID:** {retrieved_job.get('id', 'N/A')}")
    st.markdown(f"**Model:** {retrieved_job.get('model', 'N/A')}")
    st.markdown(f"**Created At:** {retrieved_job.get('created_at', 'N/A')}")
    st.markdown(f"**Finished At:** {retrieved_job.get('finished_at', 'N/A')}")
    hyperparams = retrieved_job.get("hyperparameters", {})
    st.markdown(f"**Number of Epochs:** {hyperparams.get('n_epochs', 'N/A')}")
