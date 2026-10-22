import streamlit as st
from openai import OpenAI

st.title("AI Chatbot")
st.write("Ask me any question!")

client = OpenAI()

question = st.text_input("Enter your question")

if st.button("Generate Answer"):

    if question:
        with st.spinner("Generating answer..."):

            try:
                response = client.responses.create(
                    model="YOUR_AVAILABLE_MODEL",
                    input=question
                )

                st.subheader("AI Answer")
                st.write(response.output_text)

            except Exception as error:
                st.error(f"Something went wrong: {error}")

    else:
        st.warning("Please enter a question.")
