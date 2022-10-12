import streamlit as st
from transformers import AutoTokenizer,AutoModelForQuestionAnswering
from transformers.pipelines import pipeline

st.cache(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("twmkn9/distilbert-base-uncased-squad2", return_tensors="pt")
    model = AutoModelForQuestionAnswering.from_pretrained("twmkn9/distilbert-base-uncased-squad2")
    nlp_pipe = pipeline('question-answering',model=model,tokenizer=tokenizer)
    return nlp_pipe
npl_pipe = load_model()
from PIL import Image
st.header("Solução de NLP - Meliuz")
st.text("Chat para tirar dúvidas.")
add_text_sidebar = st.sidebar.title("Menu")
add_text_sidebar = st.sidebar.text("Chat")

question = st.text_input(label='Insira uma pergunta.')
text = st.text_area(label="Resposta")
image = Image.open("C:\\Users\\pablo\\Documents\\basededados\\meliuz.png")
st.sidebar.image(image, use_column_width=True)





if (not len(text)==0) and (not len(question)==0):
    x_dict = npl_pipe(context=text,question=question)
    st.text(x_dict['answer'])