


```bash
cd meeting_trainer
python3 -m venv venv_meeting_trainer
source venv_meeting_trainer/bin/activate

pip install streamlit streamlit-mic-recorder google-generativeai

streamlit run app.py


# add secret manager

gcloud run deploy communication-copilot-ai \
  --source . \
  --region asia-northeast1 \
  --allow-unauthenticated \
  --set-secrets="GEMINI_API_KEY=GEMINI_API_KEY:latest"

deactivate
```