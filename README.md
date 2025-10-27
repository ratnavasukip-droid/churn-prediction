See files in src/, api/, dashboard/, data/.
Quick start:
1. python -m venv venv && source venv/bin/activate
2. pip install -r requirements.txt
3. python data/generate_synthetic.py
4. python src/train.py
5. uvicorn api.app:app --reload --port 8000
6. streamlit run dashboard/app_streamlit.py
