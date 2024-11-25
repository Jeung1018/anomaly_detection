import sys
import os
import streamlit as st
from datetime import datetime

# Add the parent directory of "chatbot" to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph_start.chatbot.graph import llm_app  # Import from graph.py

# Initialize session state for storing previous queries and answers
if "history" not in st.session_state:
    st.session_state.history = []  # List to store past queries and answers

# Streamlit App
st.title("Interactive Anomalies Detection")
st.write("Interact with the LangGraph workflow asking a question below.")

# Sample questions with a styled section
st.markdown(
    """
    <div style="background-color: #2e2e2e; padding: 10px; border-radius: 8px; margin-bottom: 20px;">
        <h3 style="color: #f39c12; margin-bottom: 10px;">Sample Questions:</h3>
        <ul style="color: #ffffff;">
            <li>Is there any potential short-cycling issue in our building?</li>
            <li>Can you see any short cycling trends on breaker 28722?</li>
            <li>What are the frequent reasons for the short cycling?</li>
            <li>Can I retrieve real time data?</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# User query input
query = st.text_input("Enter your query:", placeholder="Type your question here...")

# Submit Button
if st.button("Submit"):
    if query:
        try:
            with st.spinner("Processing your query..."):
                # Run the LangGraph workflow
                response = llm_app.invoke({'query': query}, {'recursion_limit': 30})

                # Extract the answer from the response
                final_answer = response.get('answer', 'No answer generated.')

                # Save the query and answer with a timestamp to session state
                st.session_state.history.append({
                    "query": query,
                    "answer": final_answer,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

                # Display the current answer
                st.success("Generated Answer:")
                st.write(final_answer)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query.")

# Display history of queries and answers
st.subheader("Chat History:")
if st.session_state.history:
    for item in st.session_state.history:
        st.markdown(f"**[{item['timestamp']}] Q: {item['query']}**")
        st.markdown(f"**A:** {item['answer']}")
        st.markdown("---")
else:
    st.info("No chat history yet.")

# Clear History Button
if st.button("Clear History"):
    st.session_state.history = []  # Clear the history
    st.info("Chat history cleared.")
