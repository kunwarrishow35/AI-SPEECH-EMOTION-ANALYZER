@echo off
echo Starting Speech Emotion Analyzer...
call .\.venv\Scripts\activate.bat
streamlit run app.py
pause
