Frontend (React + Tailwind) and Backend (FastAPI) for EyeScan

Backend (FastAPI)

- Location: `backend/api_server.py`
- Run (from project root):

```powershell
& 'k:\DL_Anemia_Jaundice\venv\Scripts\python.exe' -m pip install -r backend/requirements.txt
python backend/api_server.py
```

- API: POST /predict
  - form fields: `condition` ("jaundice" or "anemia"), `file` (image file)
  - returns JSON with probabilities, predicted label, and confidence

Frontend (React + Vite + Tailwind)

- Location: `frontend/`
- Run (from project root):

```powershell
cd frontend
npm install
npm run dev
```

- The frontend posts to `http://localhost:8000/predict` by default.

Notes

- The FastAPI backend reuses the Keras models in `models/` and `jaundice_model/models`.
- This scaffold is minimal; production security, storage, and model-serving best practices are not implemented.
