
import streamlit as st
from google.colab import userdata
from openai import OpenAI

st.title("AI Chatbot")
st.write("Ask me any question!")

try:
    # Corrected the key name to 'OPENAI_API_KEY'
    api_key = userdata.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
except Exception as error:
    st.error(f"API Key Error: {error}")
    st.info("Please add your key to Colab Secrets (🔑) under the name 'OPENAI_API_KEY' and enable notebook access.")
    client = None

question = st.text_input("Enter your question")

if st.button("Generate Answer"):
    if not client:
        st.error("OpenAI Client is not initialized. Please configure your API Key.")
    elif question:
        with st.spinner("Generating answer..."):
            try:
                # Updated to use the correct v1 API call chat.completions.create
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": question}]
                )
                st.subheader("AI Answer")
                st.write(response.choices[0].message.content)
            except Exception as error:
                st.error(f"Something went wrong: {error}")
    else:
        st.warning("Please enter a question.")
