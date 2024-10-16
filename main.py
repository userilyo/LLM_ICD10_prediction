import streamlit as st

st.set_page_config(page_title="Simple Streamlit App", layout="wide")

st.title("Simple Streamlit App")

st.write("Hello, World! This is a minimal Streamlit application.")

st.sidebar.title("About")
st.sidebar.info("This is a simple Streamlit app to test the environment.")
