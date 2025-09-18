# Curia Logica
**Where models deliberate, data testifies, and truth is entered.**

A modern, responsive **FastAPI** application that turns a plain **questions.txt** file and an optional **dataset** (CSV/XLSX/JSON/Parquet/PDF/Images/DB) into **structured answers**. Under the hood, Curia Logica convenes a *council of models* (OpenAI, Gemini, Claude), extracts tables from PDFs and images (PyMuPDF + OCR), generates **runnable Python** to compute results, and returns a clean **JSON** payload you can copy or save.

---

## ✨ Highlights

- **Multi-LLM switch**: Choose **OpenAI / Gemini / Claude** + model at request time; provide API key in the UI.
- **Attractive, responsive UI**: Fixed layout (no body scroll), **sticky** actions, scrollable result panel, and an **Analyzing…** loader.
- **Rich file intake**: CSV, XLSX, JSON, Parquet, SQLite/DuckDB, ZIP/TAR (first table), **PDF (PyMuPDF)**, **images (OCR)**.
- **Reproducible outputs**: LLM returns `{ "keys": [...], "code": "..." }`; code is executed safely; results returned as JSON.
- **Optional web table tool**: A built-in scraper tool the agent can call (when no dataset is uploaded).
- **Privacy-aware**: API keys can be entered per request (kept client-side except for that call). No long-term persistence.

---

## 🚀 Quickstart

### 1) Prerequisites
- **Python 3.10+**
- **Tesseract OCR** (for images/PDF OCR fallback)  
  - Windows (PowerShell): `choco install tesseract`  
  - macOS (Homebrew): `brew install tesseract`  
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
- (Optional) **Playwright** if you want dynamic JS page scraping by the tool: `pip install playwright && playwright install`

### 2) Install
```bash
git clone https://github.com/akshayds23/Curia-logica
cd curia-logica
python -m venv env
# Windows:
env\Scripts\activate
# macOS/Linux:
source env/bin/activate

pip install -r requirements.txt
```

### 3) Configure keys (two options)

- **Preferred (per-request)**: Enter your provider API key in the UI for each run.  
- **Environment**: Put keys in a `.env` file at project root:
```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_MODEL=gemini-2.5-pro
LLM_TIMEOUT_SECONDS=180
```

> If you don’t set `GOOGLE_API_KEY`, Gemini won’t be initialized by default—just choose OpenAI/Claude or provide the Gemini key in the UI when running.

### 4) Run
```bash
uvicorn app:app --reload
```
Open http://127.0.0.1:8000

---

## 🖥️ Using the UI (first impression guide)

1. **Provider & Model**  
   Choose **Gemini / OpenAI / Claude**. The **Model** dropdown auto-populates top models.  
   Paste your **API key** (kept in the browser for this request).

2. **Upload files**
   - **Questions (.txt)** — Required. A simple, human-readable spec of what you want computed.
   - **Dataset (optional)** — CSV/XLSX/JSON/Parquet/PDF/Images/SQLite/DuckDB/ZIP/TAR.

3. **Run Analysis**  
   Click **Run Analysis** (left panel). You’ll see **Analyzing…** in the results panel.  
   When complete, use **Copy JSON** / **Save JSON** (sticky toolbar) to export.

4. **Layout UX**
   - The **page is fixed** (no body scroll).
   - Left panel (inputs) scrolls independently; the **Run/ Clear buttons are sticky at the bottom**.
   - Right panel (results) scrolls; **Copy / Save** are sticky at the top.

---

## 🧩 What goes in `questions.txt`?

Use plain English to describe your outputs. Curia Logica prompts the LLM to **return only JSON** with:
- `keys`: array of exact output keys (you name these)
- `code`: Python that assigns a dict `results` with those keys

**Example (sales analytics):**
```
# Output keys and types:
total_sales: number
top_5_products: string[]
sales_by_region: bar_chart

# Notes:
- Treat "bar_chart" as a matplotlib plot and return base64 (<100kB) via plot_to_base64().
- If dataset not provided, you may scrape a table via the built-in tool.
```

**Another example (PDF tables):**
```
# Keys:
table_row_count: integer
first_5_rows: string
quality_check_notes: string

# Notes:
- If a PDF is uploaded, extract tables or OCR text; summarize if table extraction fails.
```

> You have full naming freedom. Keep keys short and descriptive (e.g., `summary`, `chart_sales_by_month`, `anomalies`).

---

## 🧠 End-to-End Workflow

**Narrative**
1. **UI submission** → `/api` receives `provider`, `model`, `api_key`, `questions_file`, and (optional) `data_file`.
2. **Data loading** → The server loads the dataset:
   - CSV/XLSX/JSON/Parquet/DB/Archives → **pandas**/**DuckDB**
   - **PDF** → **PyMuPDF** (text extraction + page render to image + OCR); attempts table reconstruction
   - **Images (PNG/JPG)** → **Tesseract OCR**; attempts table reconstruction
3. **Prompting** → Builds a strict prompt: *Return only JSON with `keys` and `code`*.  
4. **LLM selection** → Per request, constructs an LLM (OpenAI/Gemini/Claude) and runs the **agent**.
5. **Code generation** → The LLM returns Python in `code` that populates `results`.
6. **Safe execution** → Code is run in a sandboxed subprocess with utilities like `plot_to_base64()`.
7. **Response** → The executed `results` dict is returned as JSON and rendered in the results panel.

**Flow sketch**
```
[UI]
  └─ form-data (provider, model, api_key, questions.txt, dataset?)
       ↓
[FastAPI /api]
  ├─ Load dataset (pandas/DuckDB) or OCR/PyMuPDF
  ├─ Build strict JSON+code prompt
  ├─ Select LLM (OpenAI|Gemini|Claude)
  ├─ Agent -> { keys, code }
  ├─ Sandbox execute code -> results{}
  └─ Return JSON -> UI (Copy / Save)
```

---

## 🧪 Programmatic API (optional)

`POST /api` — **multipart/form-data**

- `provider` (str) – `openai | gemini | claude` (default: `gemini`)
- `model` (str) – model id/name (optional; sensible default used per provider)
- `api_key` (str) – per-request key (optional; or use `.env`)
- `questions_file` (file, required) – `.txt`
- `data_file` (file, optional) – dataset (csv/xlsx/json/parquet/pdf/png/jpg/db/zip/tar…)

**cURL example**
```bash
curl -X POST http://127.0.0.1:8000/api   -F provider=openai   -F model=gpt-4o-mini   -F api_key="$OPENAI_API_KEY"   -F questions_file=@questions.txt   -F data_file=@sales.csv
```

**Response (shape)**
```json
{
  "total_sales": 123456.78,
  "top_5_products": ["A", "B", "C", "D", "E"],
  "sales_by_region": "data:image/png;base64,iVBORw0KG..."
}
```

---

## 🔐 Security & Privacy

- **API keys**: The UI can send keys per request; the server **does not store** them.
- **Files**: Read in-memory; short-lived temp files for execution; cleaned up after each run.
- **No external calls** unless you omit a dataset and your questions instruct the agent to use the built-in scraper.
- **Determinism**: Temperature is set to **0** to reduce variance.

---

## 🧰 Troubleshooting

**`google.auth.exceptions.DefaultCredentialsError` at startup**  
You started without `GOOGLE_API_KEY`. Either:
- Enter a key per request in the UI (choose **Gemini**).
- Or set `GOOGLE_API_KEY` in `.env`.  
(If not provided, use **OpenAI** or **Claude**, which don’t rely on Google ADC.)

**Tesseract not found**  
Install it and ensure it’s on your PATH (see prerequisites above).

**Playwright not installed**  
Dynamic JS rendering in the scraper won’t run. Install:  
`pip install playwright && playwright install`

**Large images/plots exceed 100kB**  
The runner’s `plot_to_base64()` helper **auto-downscales**; keep figures simple and text minimal.

**Windows PATH / venv issues**  
Use the project’s virtual environment (`env\Scripts\activate`) before running `uvicorn`.

---

## 🗂️ Project Structure

```
.
├─ app.py                # FastAPI app, multi-LLM agent, OCR/PyMuPDF, safe code runner
├─ index.html            # Responsive, modern UI (sticky actions, scrollable results)
├─ requirements.txt      # Python dependencies
├─ favicon.ico           # App icon (optional)
└─ README.mfd            # This file
```

---

## 🛣️ Roadmap (suggested)

- Multi-table PDF extraction & schema unification
- Pluggable evaluation suite (golden questions + assertions)
- Streaming partial results & live logs in UI
- Named prompt templates per domain
- Role-based policy checks (PII/PHI guards)

---

## 🤝 Contributing

1. Fork → create a feature branch.
2. Keep commits focused and documented.
3. Open a PR with a clear before/after in the description.
4. Please include repro steps or sample `questions.txt` + small CSV.

---

## 📄 License

MIT (suggested). Update `LICENSE` to your preference.

---

## 🧭 Brand notes

- **Name**: **Curia Logica** — a deliberative chamber for reason.
- **Tagline**: *An analytic curia for modern data.*
- **Type system**: Pair a classic serif (e.g., EB Garamond) with a refined script (e.g., Tangerine) for headers if desired.
