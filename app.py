import streamlit as st
from streamlit_mic_recorder import mic_recorder
import google.generativeai as genai
import io, os

# 1. Gemini API Configuration
# Replace 'YOUR_GEMINI_API_KEY' with your actual key from Google AI Studio
genai.configure(api_key = os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

st.set_page_config(page_title="Communication Copilot AI", page_icon="🎙️", layout="centered")

# --- UI Header ---
st.title("🎙️ AI Communication Copilot AI")
st.subheader("Your Real-time Communication Co-Pilot")
st.markdown("""
Elevate your professional presence. Record your speech during meetings 
to receive instant feedback on clarity, tone, and professionalism.
""")

st.divider()

# --- Recording Section ---
st.write("Click the button below to start recording your speech:")
audio = mic_recorder(
    start_prompt="Start Recording ⏺️",
    stop_prompt="Stop & Analyze 📤",
    key='recorder'
)

# --- Processing & Analysis ---
if audio:
    # Playback for user confirmation
    st.audio(audio['bytes']) 
    
    with st.spinner('Analyzing your communication patterns...'):
        try:
            # Prepare audio data for Gemini
            audio_data = {
                "mime_type": "audio/wav",
                "data": audio['bytes']
            }
            
            # Professional Persona-based Prompt
            prompt = """
            You are an expert Executive Communication Coach. 
            Analyze the provided audio and provide feedback in English:
            
            1. **Executive Summary**: Briefly summarize the key points discussed.
            2. **Professional Polish**: Suggest more sophisticated or professional 
               phrases for what was said (e.g., instead of "I think", use "I am confident that").
            3. **Speech Clarity**: Identify filler words (um, uh, like) and suggest 
               improvements for tone and pacing.
            4. **Action Items**: List any next steps or commitments mentioned.
            
            Format the output with clear headings and bullet points.
            """
            
            # Call Gemini Model
            response = model.generate_content([prompt, audio_data])
            
            # Display Results
            st.success("✅ Analysis Complete!")
            st.markdown("### 📊 Feedback Report")
            st.markdown(response.text)
            
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

# --- Footer / Tips ---
st.info("Tip: Aim for 1-2 minute updates.")