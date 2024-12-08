import streamlit as st
import joblib



# Load the trained model and vectorizer
MODEL = joblib.load('spam_detection_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# App configuration for a polished design
st.set_page_config(page_title="Email Spam Detection", page_icon="📧", layout="wide")

# App Header with Logo
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://i.ibb.co.com/6gK9x0p/logo.jpg" alt="Project Logo" style="border-radius: 10px; width: 100px; height: 100px;">
        <h1 style="color: #4CAF50; margin-top: 20px;">📧 Email Spam Detection App</h1>
        <p>Classify your email content as <strong>Spam</strong> or <strong>Ham</strong> with ease!</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# Sidebar with project details
with st.sidebar:
    st.title("📋 Project Details")
    st.markdown(
        """
        - **Project Name**: Email Spam Detection  
        - **Developed By**: Mominur Rahman  
        - **GitHub**: [email-spam-detection](https://github.com/mominurr/Email-Spam-Detection-With-Machine-Learning)  
        - **Portfolio**: [mominur.dev](https://mominur.dev)
        - **About**: An intuitive Email Spam Detection app that uses machine learning to classify email content as Spam or Ham with ease and accuracy.
        """
    )
    st.image("https://i.ibb.co.com/6gK9x0p/logo.jpg")

# State management for clear/reset functionality
if "result" not in st.session_state:
    st.session_state["result"] = None
if "email_text" not in st.session_state:
    st.session_state["email_text"] = ""


# Main content area
st.subheader("🔍 Analyze Your Email")
email_text = st.text_area(
    "✏️ Enter Email Text Below",
    st.session_state["email_text"],
    placeholder="Type or paste your email content here...",
    key = "email_input"
)

# Buttons for actions
col_buttons = st.columns(7, gap="small")
with col_buttons[0]:
    if st.button("🔮 Predict", key="predict_btn"):
        if email_text.strip():
            # Vectorize input text and predict
            input_text_vectorized = vectorizer.transform([email_text])
            prediction = MODEL.predict(input_text_vectorized)

            # Determine result
            result = "Ham (Not Spam)" if prediction[0] == 0 else "Spam"
            result_color = "green" if prediction[0] == 0 else "red"
            emoji = "✅" if prediction[0] == 0 else "🚨"

            # Store results in session state
            st.session_state["result"] = (emoji, result, result_color)
            st.session_state["email_text"] = email_text
        else:
            st.error("Please enter valid email text to analyze.")
with col_buttons[1]:
    if st.button("🗑️ Clear", key="clear_btn"):
        # Reset session state to clear fields
        st.session_state["result"] = None
        st.session_state["email_text"] = ""


# Display results if available
if st.session_state["result"]:
    emoji, result, result_color = st.session_state["result"]
    st.markdown(f"### {emoji} **The email is classified as:**")
    st.markdown(f"<h2 style='color:{result_color};'>{result}</h2>", unsafe_allow_html=True)

# Footer with credits
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 14px; color: grey;">
        Developed by <a href="https://mominur.dev" target="_blank" style="color: #4CAF50;">Mominur Rahman</a>
    </div>
    """,
    unsafe_allow_html=True,
)
