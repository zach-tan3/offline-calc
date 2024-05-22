import streamlit as st
import replicate
import os
import pandas as pd
import torch
import numpy as np
from io import BytesIO
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")
st.session_state.last_prediction_probability = " "

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    st.write('This chatbot is created using the open-source Llama 2 LLM model from Meta.')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader('Models and parameters')
    selected_model = st.selectbox('Choose a Llama2 model', ['Llama2-70B', 'Llama2-13B', 'Llama2-7B'], key='selected_model')
    if selected_model == 'Llama2-70B':
        llm = 'replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48'
    elif selected_model == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    elif selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    temperature = st.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.slider('max_length', min_value=32, max_value=9999, value=120, step=8)
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')

# Create an instance of the model
model_path = "discriminator_final"  # Path to your model file in the GitHub repository
model = Discriminator(input_size=14)  # Assuming the input size is 6, you need to update it accordingly
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "This is a risk calculator for need for of admission into an Intensive Care Unit (ICU) of a paitent post-surgery. Ask me anything"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input, llm):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'. "
    string_dialogue = "You are a helpful healthcare assistant designed to aid users with healthcare-related questions. You are not a substitute for professional medical advice. Always consult a healthcare provider for medical concerns. You only respond once as 'Assistant'.\n\n"
    string_dialogue += "You are part of a project that aims to revolutionize healthcare by leveraging data science and Generative AI technologies to improve patient care and optimize clinical workflows. By integrating Generative AI, the goal is to create a cutting-edge framework capable of autonomously generating a wide range of rich and diverse content, including text, images, and other media types. Our primary focus is on creating a risk calculator to predict mortality and the need for intensive care unit (ICU) stay using data analytics and Meta AI Technologies."
    string_dialogue += "You are to give the last predicted probability from your chat history of need for ICU stay if asked and explain that the predicted probability is the probability of need for ICU stay after a surgery."
    string_dialogue += "The following is a data dictionary of an explanation of each variable which you are to explain to the user if asked: \n"
    string_dialogue += "AGE: Age\n"
    string_dialogue += "GENDER: Gender\n"
    string_dialogue += "RCRIScore: Revised Cardiac Risk Index, see [Wikipedia](https://en.wikipedia.org/wiki/Revised_Cardiac_Risk_Index)\n"
    string_dialogue += "AnemiaCategory: Based on concentration of haemoglobin as per WHO guidelines. May be None, Mild, Moderate, Severe\n"
    string_dialogue += "PreopEGFRMDRD: EGFR = estimated glomerular filtration rate. MDRD = Modification of Diet in Renal Disease equation. Measure of pre-exisiting kidney disease.\n"
    string_dialogue += "GradeofKidneyDisease: Classification of kidney disease statsus based on GFR (see above): see [Kidney.org](https://www.kidney.org/professionals/explore-your-knowledge/how-to-classify-ckd)\n"
    string_dialogue += "AnaesthesiaTypeCategory: General or Regional anaesthesia\n"
    string_dialogue += "PriorityCategory: Elective or Emergency surgery (Emregency = must be done within 24 hours)\n"
    string_dialogue += "AGEcategory: Categorisation of age\n"
    string_dialogue += "SurgicalRiskCategory: Surgical Risk Category (may be low, High, Moderate). Based on based on the 2014 European Society of Cardiology (ESC) and the European Society of Anaesthesiology (ESA) guidelines\n"
    string_dialogue += "RaceCategory: Race\n"
    string_dialogue += "AnemiaCategoryBinned: See #5; Moderate and Severe combined\n"
    string_dialogue += "RDW157: Red Cell Distribution Width > 15.7%\n"
    string_dialogue += "ASACategoryBinned: Surgical risk category, based on ASA-PS. [ASA-PS](https://www.asahq.org/standards-and-practice-parameters/statement-on-asa-physical-status-classification-system)\n"
    
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run(llm, 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt, llm)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)

Age = st.sidebar.slider('Age', 18, 99, 40)
Gender = st.sidebar.selectbox('Gender', ['female', 'male'])
RCRIScore = st.sidebar.select_slider('RCRIScore', options=[0, 1, 2, 3, 4, 5])
AnemiaCategory = st.sidebar.selectbox('Anemia Category', ['none', 'mild', 'moderate', 'severe'])
PreopEGFRMDRD = st.sidebar.slider('PreopEGFRMDRD', 0, 160, 80)
GradeofKidneyDisease = st.sidebar.selectbox('Grade of Kidney Disease', ['blank', 'g1', 'g2', 'g3a', 'g3b', 'g4', 'g5'])
AnesthesiaTypeCategory = st.sidebar.selectbox('Anaestype', ['ga', 'ra'])
PriorityCategory = st.sidebar.selectbox('Priority', ['elective', 'emergency'])
SurgicalRiskCategory = st.sidebar.selectbox('SurgRisk', ['low', 'moderate', 'high'])
RaceCategory = st.sidebar.selectbox('Race', ['chinese', 'indian', 'malay', 'others'])
AnemiaCategoryBinned = st.sidebar.selectbox('Anemia Category Binned', ['none', 'mild', 'moderate/severe'])
RDW157 = st.sidebar.selectbox('RDW15.7', ['<= 15.7', '>15.7'])
ASACategoryBinned = st.sidebar.selectbox('ASA Category Binned', ['i', 'ii', 'iii', 'iv-vi'])

age_category = None
if Age < 30:
    age_category = '18-29'
elif Age < 50:
    age_category = '30-49'
elif Age < 65:
    age_category = '50-64'
elif Age < 75:
    age_category = '65-74'
elif Age < 85:
    age_category = '75-84'
else:
    age_category = '>=85'

prediction_prompt = {'Age': Age,
                     'Gender': Gender, 
                     'RCRIScore': RCRIScore,
                     'AnemiaCategory': AnemiaCategory, 
                     'PreopEGFRMDRD': PreopEGFRMDRD, 
                     'GradeofKidneyDisease': GradeofKidneyDisease, 
                     'AnaesthesiaTypeCategory': AnesthesiaTypeCategory, 
                     'PriorityCategory': PriorityCategory, 
                     'AgeCategory': age_category, 
                     'SurgicalRiskCategory': SurgicalRiskCategory, 
                     'RaceCategory': RaceCategory, 
                     'AnemiaCategoryBinned': AnemiaCategoryBinned, 
                     'RDW15.7': RDW157, 
                     'ASACategoryBinned': ASACategoryBinned}

if st.sidebar.button('Predict'):
    with st.chat_message("user"):
        st.write(prediction_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Preprocess your input data
            input_data = pd.DataFrame({'Age': [Age],
                                       'Gender': [Gender],
                                       'RCRIScore': [RCRIScore],
                                       'AnemiaCategory': [AnemiaCategory],
                                       'PreopEGFRMDRD': [PreopEGFRMDRD],
                                       'GradeofKidneyDisease': [GradeofKidneyDisease],
                                       'AnesthesiaTypeCategory': [AnesthesiaTypeCategory],
                                       'PriorityCategory': [PriorityCategory],
                                       'AgeCategory': [age_category],
                                       'SurgicalRiskCategory': [SurgicalRiskCategory],
                                       'RaceCategory': [RaceCategory],
                                       'AnemiaCategoryBinned': [AnemiaCategoryBinned],
                                       'RDW15.7': [RDW157],
                                       'ASACategoryBinned': [ASACategoryBinned]})

            # Mappings of categorical values
            gender_mapper = {"female": 0, "male": 1}
            anemia_category_mapper = {"none":0, "mild":1, "moderate":2, "severe":3}
            GradeofKidneydisease_mapper = {"blank":0, "g1":1, "g2":2, "g3a":3,"g3b":4, "g4":5, "g5":6}
            anaestype_mapper = {"ga": 0, "ra": 1}
            priority_mapper = {"elective": 0, "emergency": 1}
            AGEcategory_mapper = {"18-29":0, "30-49":1, "50-64":2,"65-74":3, "75-84":4, ">=85":5}
            SurgRiskCategory_mapper = {"low":0, "moderate":1, "high":2}
            race_mapper = {"chinese": 0, "indian": 1, "malay": 2, "others": 3}
            Anemiacategorybinned_mapper = {"none": 0, "mild":1, "moderate/severe":2}
            RDW157_mapper = {"<= 15.7":0, ">15.7":1}
            ASAcategorybinned_mapper = {"i":0, "ii":1, 'iii':2, 'iv-vi':3}
            
            # Map categorical values
            input_data['Gender'] = input_data['Gender'].map(gender_mapper)
            input_data['AnemiaCategory'] = input_data['AnemiaCategory'].map(anemia_category_mapper)
            input_data['GradeofKidneyDisease'] = input_data['GradeofKidneyDisease'].map(GradeofKidneydisease_mapper)
            input_data['AnesthesiaTypeCategory'] = input_data['AnesthesiaTypeCategory'].map(anaestype_mapper)
            input_data['PriorityCategory'] = input_data['PriorityCategory'].map(priority_mapper)
            input_data['AgeCategory'] = input_data['AgeCategory'].map(AGEcategory_mapper)
            input_data['SurgicalRiskCategory'] = input_data['SurgicalRiskCategory'].map(SurgRiskCategory_mapper)
            input_data['RaceCategory'] = input_data['RaceCategory'].map(race_mapper)
            input_data['AnemiaCategoryBinned'] = input_data['AnemiaCategoryBinned'].map(Anemiacategorybinned_mapper)
            input_data['RDW15.7'] = input_data['RDW15.7'].map(RDW157_mapper)
            input_data['ASACategoryBinned'] = input_data['ASACategoryBinned'].map(ASAcategorybinned_mapper)

            # Convert to PyTorch tensor
            input_tensor = torch.tensor(input_data.values, dtype=torch.float32)

            # Generate prediction
            with torch.no_grad():
                probability = model(input_tensor)
                predicted = (probability >= 0.5).float()  # Here, you are using a threshold of 0.5 to determine the class.

            # Save prediction probability
            st.session_state.last_prediction_probability = f"Predicted probability: {probability.item() * 100:.2f}%"
            
            # Display prediction
            st.write(st.session_state.last_prediction_probability)

            message = {"role": "assistant", "content": st.session_state.last_prediction_probability}
            st.session_state.messages.append(message)
