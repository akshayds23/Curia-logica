import os
import re
import io
import json
import base64
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import duckdb
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
from dotenv import load_dotenv

# Lightweight Excel reader (optional but helps keep "no compromise" for xlsx)
try:
    import openpyxl  # type: ignore
    HAS_OPENPYXL = True
except Exception:
    HAS_OPENPYXL = False

# Optional image handling (we only need it to validate/normalize before OCR, not for OCR itself)
try:
    from PIL import Image  # type: ignore
    HAS_PIL = True
except Exception:
    HAS_PIL = False

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("curia-logica")

app = FastAPI(title="Curia Logica - Data Analyst Agent (Serverless-friendly)")

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "180"))
EXEC_TIMEOUT_SECONDS = int(os.getenv("EXEC_TIMEOUT_SECONDS", "120"))

# OCR.space key MUST be set in Vercel env for OCR features
OCRSPACE_API_KEY = os.getenv("OCRSPACE_API_KEY", "").strip()

# ---------------------------------------------------------------------
# Model allow-lists (no defaults; user must pick one)
# ---------------------------------------------------------------------

OPENAI_MODELS: List[str] = [
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5.1",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o4-mini",
    "o3",
]

ANTHROPIC_MODELS: List[str] = [
    "claude-opus-4-6",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
]

GEMINI_MODELS: List[str] = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-2.5-flash-lite-preview-09-2025",
]

PROVIDER_MODELS: Dict[str, List[str]] = {
    "openai": OPENAI_MODELS,
    "anthropic": ANTHROPIC_MODELS,
    "gemini": GEMINI_MODELS,
}

# ---------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    base_dir = Path(__file__).resolve().parent
    html_path = base_dir / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            content="<h1>Frontend not found</h1><p>index.html is missing.</p>",
            status_code=404,
        )
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------
# Helpers: JSON extraction
# ---------------------------------------------------------------------
def clean_llm_json(output: str) -> Dict[str, Any]:
    if not output:
        return {"error": "Empty LLM output"}

    s = output.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    first = s.find("{")
    last = s.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return {"error": "No JSON object found in LLM output", "raw": s}

    candidate = s[first:last + 1]
    try:
        return json.loads(candidate)
    except Exception as e:
        return {"error": f"JSON parse failed: {e}", "raw": candidate}

# ---------------------------------------------------------------------
# OCR.space (PDF/JPG/PNG) -> text
# ---------------------------------------------------------------------
def ocr_space_extract_text(file_bytes: bytes, filename: str) -> str:
    if not OCRSPACE_API_KEY:
        raise HTTPException(400, "OCRSPACE_API_KEY is missing in environment variables.")

    url = "https://api.ocr.space/parse/image"
    files = {"file": (filename, file_bytes)}
    data = {"apikey": OCRSPACE_API_KEY, "language": "eng", "isOverlayRequired": "false"}

    try:
        resp = requests.post(url, files=files, data=data, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        raise HTTPException(502, f"OCR.space request failed: {e}")

    if payload.get("IsErroredOnProcessing"):
        err = payload.get("ErrorMessage") or payload.get("ErrorDetails") or "Unknown OCR error"
        raise HTTPException(400, f"OCR.space error: {err}")

    parsed = payload.get("ParsedResults") or []
    if not parsed:
        return ""

    text_parts = []
    for item in parsed:
        t = item.get("ParsedText", "")
        if t:
            text_parts.append(t)

    return "\n".join(text_parts).strip()

def lines_to_table_rows(lines: List[str]) -> Optional[List[List[str]]]:
    rows: List[List[str]] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        parts = re.split(r"\s{2,}", s)
        if len(parts) >= 2:
            rows.append(parts)

    if not rows:
        return None

    counts: Dict[int, int] = {}
    for r in rows:
        counts[len(r)] = counts.get(len(r), 0) + 1

    best_cols = max(counts, key=counts.get)
    if best_cols < 2:
        return None

    filtered = [r for r in rows if len(r) == best_cols]
    if len(filtered) < max(3, int(0.4 * len(rows))):
        return None

    return filtered

def write_rows_to_csv(rows: List[List[str]]) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    path = tmp.name

    header = rows[0]

    def _is_number(x: str) -> bool:
        try:
            float(x.replace(",", ""))
            return True
        except Exception:
            return False

    header_non_numeric = sum(1 for c in header if not _is_number(c))
    header_unique = len(set(header)) == len(header)
    has_header = header_unique and header_non_numeric >= max(1, len(header) // 2)

    import csv
    w = csv.writer(tmp)
    if has_header:
        w.writerow([c.strip() for c in header])
        for r in rows[1:]:
            w.writerow(r)
    else:
        w.writerow([f"col_{i+1}" for i in range(len(rows[0]))])
        for r in rows:
            w.writerow(r)

    tmp.flush()
    tmp.close()
    return path

# ---------------------------------------------------------------------
# QuickChart -> base64 PNG
# ---------------------------------------------------------------------
def quickchart_to_base64(chart_config: dict, width: int = 700, height: int = 400) -> str:
    try:
        resp = requests.post(
            "https://quickchart.io/chart",
            json={"chart": chart_config, "width": width, "height": height, "format": "png"},
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(502, f"QuickChart request failed: {e}")

    return base64.b64encode(resp.content).decode("utf-8")

# ---------------------------------------------------------------------
# Dataset loading into DuckDB (no pandas)
# ---------------------------------------------------------------------
def save_upload_to_temp(content: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return tmp.name

def xlsx_to_csv_temp(xlsx_path: str) -> str:
    if not HAS_OPENPYXL:
        raise HTTPException(400, "openpyxl is required to read .xlsx files (add it to requirements).")

    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    ws = wb.worksheets[0]

    import csv
    out = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    out_path = out.name
    w = csv.writer(out)

    for row in ws.iter_rows(values_only=True):
        w.writerow(["" if v is None else v for v in row])

    out.flush()
    out.close()
    return out_path

def duckdb_load_table(
    conn: duckdb.DuckDBPyConnection,
    file_path: str,
    original_name: str
) -> Tuple[str, List[str], List[Tuple[Any, ...]]]:
    name = (original_name or "").lower()
    table = "data"
    conn.execute("DROP TABLE IF EXISTS data")

    if name.endswith(".csv"):
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto(?, HEADER=true)", [file_path])
    elif name.endswith(".parquet"):
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM read_parquet(?)", [file_path])
    elif name.endswith(".json"):
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM read_json_auto(?)", [file_path])
    elif name.endswith(".db") or name.endswith(".sqlite") or name.endswith(".sqlite3"):
        conn.execute("INSTALL sqlite; LOAD sqlite;")
        conn.execute("ATTACH ? AS uploaded_db (TYPE sqlite)", [file_path])
        tables = conn.execute("SHOW TABLES FROM uploaded_db").fetchall()
        if not tables:
            raise HTTPException(400, "No tables found in SQLite DB.")
        first_table = tables[0][0]
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM uploaded_db.{first_table}")
    elif name.endswith(".duckdb"):
        conn.execute("ATTACH ? AS uploaded_db", [file_path])
        tables = conn.execute("SHOW TABLES FROM uploaded_db").fetchall()
        if not tables:
            raise HTTPException(400, "No tables found in DuckDB file.")
        first_table = tables[0][0]
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM uploaded_db.{first_table}")
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        if name.endswith(".xls"):
            raise HTTPException(400, "'.xls' not supported in lightweight mode. Use .xlsx or convert to CSV.")
        csv_path = xlsx_to_csv_temp(file_path)
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto(?, HEADER=true)", [csv_path])
    elif name.endswith(".txt"):
        conn.execute("CREATE TABLE data(text VARCHAR)")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            conn.execute("INSERT INTO data VALUES (?)", [f.read()])
    else:
        raise HTTPException(400, f"Unsupported data file type: {original_name}")

    cols = [r[0] for r in conn.execute(f"DESCRIBE {table}").fetchall()]
    preview = conn.execute(f"SELECT * FROM {table} LIMIT 5").fetchall()
    return table, cols, preview

def format_preview_md(cols: List[str], rows: List[Tuple[Any, ...]]) -> str:
    if not cols:
        return ""

    def _cell(v: Any) -> str:
        s = "" if v is None else str(v)
        s = s.replace("\n", " ")
        if len(s) > 60:
            s = s[:57] + "..."
        return s

    header = "| " + " | ".join([_cell(c) for c in cols]) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body_lines = []
    for r in rows:
        body_lines.append("| " + " | ".join([_cell(v) for v in r]) + " |")
    return "\n".join([header, sep] + body_lines)

# ---------------------------------------------------------------------
# Web scraping helper (server-side, non-executor) -> temp file
# ---------------------------------------------------------------------
def scrape_url_to_tempfile(url: str) -> Tuple[str, str]:
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.google.com"}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch URL: {e}")

    ctype = (r.headers.get("Content-Type") or "").lower()
    lower = url.lower()

    if "application/json" in ctype or lower.endswith(".json"):
        return save_upload_to_temp(r.content, ".json"), "scraped.json"
    if "text/csv" in ctype or lower.endswith(".csv"):
        return save_upload_to_temp(r.content, ".csv"), "scraped.csv"
    if "parquet" in ctype or lower.endswith(".parquet"):
        return save_upload_to_temp(r.content, ".parquet"), "scraped.parquet"

    # HTML -> parse first table -> CSV (fallback to .txt)
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(r.text, "lxml")
        table = soup.find("table")
        if not table:
            return save_upload_to_temp(r.text.encode("utf-8", errors="ignore"), ".txt"), "scraped.txt"

        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if not cells:
                continue
            rows.append([c.get_text(" ", strip=True) for c in cells])

        if not rows:
            return save_upload_to_temp(r.text.encode("utf-8", errors="ignore"), ".txt"), "scraped.txt"

        csv_path = write_rows_to_csv(rows)
        return csv_path, "scraped.csv"
    except Exception:
        return save_upload_to_temp(r.text.encode("utf-8", errors="ignore"), ".txt"), "scraped.txt"

# ---------------------------------------------------------------------
# LLM calls (no defaults; provider+model+api_key required)
# ---------------------------------------------------------------------
def validate_provider_model(provider: str, model: str) -> None:
    provider = (provider or "").strip().lower()
    model = (model or "").strip()

    if provider not in PROVIDER_MODELS:
        raise HTTPException(400, f"Unsupported provider '{provider}'. Choose one of: {list(PROVIDER_MODELS.keys())}")

    allowed = PROVIDER_MODELS[provider]
    if model not in allowed:
        raise HTTPException(400, f"Unsupported model '{model}' for provider '{provider}'. Allowed models: {allowed}")

def call_openai(model: str, api_key: str, prompt: str) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        raise HTTPException(500, "openai package not installed.")

    client = OpenAI(api_key=api_key)
    try:
        resp = client.responses.create(model=model, input=prompt)
        text = getattr(resp, "output_text", None)
        if text:
            return text
        out = resp.output or []
        parts = []
        for item in out:
            content = item.get("content") if isinstance(item, dict) else None
            if not content:
                continue
            for c in content:
                if isinstance(c, dict) and c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
        return "\n".join(parts).strip()
    except Exception as e:
        raise HTTPException(502, f"OpenAI API call failed: {e}")

def call_anthropic(model: str, api_key: str, prompt: str) -> str:
    try:
        import anthropic  # type: ignore
    except Exception:
        raise HTTPException(500, "anthropic package not installed.")

    client = anthropic.Anthropic(api_key=api_key)
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = []
        for block in (msg.content or []):
            t = getattr(block, "text", None)
            if t:
                parts.append(t)
        return "\n".join(parts).strip()
    except Exception as e:
        raise HTTPException(502, f"Anthropic API call failed: {e}")

def call_gemini(model: str, api_key: str, prompt: str) -> str:
    try:
        from google import genai  # google-genai
    except Exception:
        raise HTTPException(500, "google-genai package not installed. Add 'google-genai' to requirements.txt")

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(model=model, contents=prompt)

        text = getattr(resp, "text", None)
        if text:
            return text.strip()

        # fallback
        parts = []
        for cand in (getattr(resp, "candidates", None) or []):
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in (getattr(content, "parts", None) or []):
                t = getattr(part, "text", None)
                if t:
                    parts.append(t)
        return "\n".join(parts).strip()
    except Exception as e:
        raise HTTPException(502, f"Gemini API call failed: {e}")

def call_llm(provider: str, model: str, api_key: str, prompt: str) -> str:
    provider = provider.lower().strip()
    validate_provider_model(provider, model)

    if not api_key.strip():
        raise HTTPException(400, "api_key is required (no default LLM configured).")

    if provider == "openai":
        return call_openai(model=model, api_key=api_key, prompt=prompt)
    if provider == "anthropic":
        return call_anthropic(model=model, api_key=api_key, prompt=prompt)
    if provider == "gemini":
        return call_gemini(model=model, api_key=api_key, prompt=prompt)

    raise HTTPException(400, f"Unsupported provider: {provider}")

# ---------------------------------------------------------------------
# Safe execution: run generated python in a subprocess
# ---------------------------------------------------------------------
EXEC_PREAMBLE = r"""
import os, json, re, base64
import duckdb
import requests

DATA_PATH = os.environ.get("DATA_PATH", "")
DATA_NAME = os.environ.get("DATA_NAME", "").lower()

def quickchart_to_base64(chart_config: dict, width: int = 700, height: int = 400) -> str:
    resp = requests.post(
        "https://quickchart.io/chart",
        json={"chart": chart_config, "width": width, "height": height, "format": "png"},
        timeout=30,
    )
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode("utf-8")

def scrape_url_to_tempfile(url: str) -> tuple[str, str]:
    headers = {"User-Agent":"Mozilla/5.0","Referer":"https://www.google.com"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    lower = url.lower()

    import tempfile
    def _save(b: bytes, suf: str) -> str:
        f = tempfile.NamedTemporaryFile(delete=False, suffix=suf)
        f.write(b); f.flush(); f.close()
        return f.name

    if "application/json" in ctype or lower.endswith(".json"):
        return _save(r.content, ".json"), "scraped.json"
    if "text/csv" in ctype or lower.endswith(".csv"):
        return _save(r.content, ".csv"), "scraped.csv"
    if "parquet" in ctype or lower.endswith(".parquet"):
        return _save(r.content, ".parquet"), "scraped.parquet"

    # HTML fallback: store as .txt
    return _save(r.text.encode("utf-8", errors="ignore"), ".txt"), "scraped.txt"

def load_into_duckdb(path: str, name: str, table: str = "data") -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(database=":memory:")
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    name = (name or "").lower()

    if not path:
        return conn

    if name.endswith(".csv"):
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto(?, HEADER=true)", [path])
    elif name.endswith(".parquet"):
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM read_parquet(?)", [path])
    elif name.endswith(".json"):
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM read_json_auto(?)", [path])
    elif name.endswith(".duckdb"):
        conn.execute("ATTACH ? AS uploaded_db", [path])
        tables = conn.execute("SHOW TABLES FROM uploaded_db").fetchall()
        if not tables:
            raise RuntimeError("No tables in DuckDB file.")
        first_table = tables[0][0]
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM uploaded_db.{first_table}")
    elif name.endswith(".db") or name.endswith(".sqlite") or name.endswith(".sqlite3"):
        conn.execute("INSTALL sqlite; LOAD sqlite;")
        conn.execute("ATTACH ? AS uploaded_db (TYPE sqlite)", [path])
        tables = conn.execute("SHOW TABLES FROM uploaded_db").fetchall()
        if not tables:
            raise RuntimeError("No tables in SQLite DB.")
        first_table = tables[0][0]
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM uploaded_db.{first_table}")
    elif name.endswith(".txt"):
        conn.execute(f"CREATE TABLE {table}(text VARCHAR)")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            conn.execute(f"INSERT INTO {table} VALUES (?)", [f.read()])
    else:
        raise RuntimeError(f"Unsupported dataset type in executor: {name}")

    return conn

duckdb_conn = load_into_duckdb(DATA_PATH, DATA_NAME, table="data")

def scrape_url_to_duckdb_table(url: str, table: str = "data") -> str:
    # Downloads URL and loads into DuckDB table.
    # Supports CSV/JSON/Parquet; HTML falls back to text in a one-column table.
    # Returns the table name.
    path, name = scrape_url_to_tempfile(url)
    name = (name or "").lower()

    duckdb_conn.execute(f"DROP TABLE IF EXISTS {table}")

    if name.endswith(".csv"):
        duckdb_conn.execute(f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto(?, HEADER=true)", [path])
    elif name.endswith(".parquet"):
        duckdb_conn.execute(f"CREATE TABLE {table} AS SELECT * FROM read_parquet(?)", [path])
    elif name.endswith(".json"):
        duckdb_conn.execute(f"CREATE TABLE {table} AS SELECT * FROM read_json_auto(?)", [path])
    elif name.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        duckdb_conn.execute(f"CREATE TABLE {table}(text VARCHAR)")
        duckdb_conn.execute(f"INSERT INTO {table} VALUES (?)", [txt])
    else:
        raise RuntimeError(f"Unsupported scraped type: {name}")

    return table

results = {}
"""


BLOCKED_IMPORT_RE = re.compile(
    r"(^|\n)\s*(import\s+(pandas|numpy|matplotlib|seaborn|pyarrow)\b|from\s+(pandas|numpy|matplotlib|seaborn|pyarrow)\s+import\b)",
    re.IGNORECASE
)

def run_user_code(code: str, data_path: str, data_name: str, timeout: int) -> Dict[str, Any]:
    # hard block heavy libs if LLM sneaks them in
    if BLOCKED_IMPORT_RE.search(code or ""):
        return {"status": "error", "message": "Generated code tried to import blocked libraries (pandas/numpy/matplotlib/pyarrow/seaborn)."}

    script = EXEC_PREAMBLE + "\n\n" + code + "\n\n" + r"""
print(json.dumps({"status":"success","result":results}, default=str), flush=True)
"""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    tmp.write(script)
    tmp.flush()
    tmp.close()

    env = os.environ.copy()
    env["DATA_PATH"] = data_path or ""
    env["DATA_NAME"] = data_name or ""

    try:
        completed = subprocess.run(
            ["python", tmp.name],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if completed.returncode != 0:
            return {
                "status": "error",
                "message": (completed.stderr.strip() or completed.stdout.strip() or "Unknown execution error"),
            }

        out = completed.stdout.strip()
        try:
            parsed = json.loads(out)
        except Exception as e:
            return {"status": "error", "message": f"Could not parse executor JSON: {e}", "raw": out}

        if parsed.get("status") != "success":
            return {"status": "error", "message": "Executor failed", "raw": parsed}

        return parsed.get("result", {})
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out"}
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

# ---------------------------------------------------------------------
# API
# ---------------------------------------------------------------------
@app.get("/api/models", include_in_schema=False)
async def list_models():
    return JSONResponse({"providers": {"openai": OPENAI_MODELS, "anthropic": ANTHROPIC_MODELS, "gemini": GEMINI_MODELS}})

@app.post("/api")
async def analyze_data(request: Request):
    form = await request.form()

    provider = (form.get("provider") or "").strip().lower()
    model = (form.get("model") or "").strip()
    api_key = (form.get("api_key") or "").strip()

    questions_file = None
    data_file = None

    for _, val in form.items():
        if hasattr(val, "filename") and getattr(val, "filename"):
            fname = val.filename.lower()
            if fname.endswith(".txt") and questions_file is None:
                questions_file = val
            else:
                data_file = val

    if not questions_file:
        raise HTTPException(400, "Missing questions file (.txt).")

    validate_provider_model(provider, model)
    if not api_key:
        raise HTTPException(400, "api_key is required (no default LLM configured).")

    raw_questions = (await questions_file.read()).decode("utf-8", errors="ignore").strip()

    dataset_uploaded = False
    data_path = ""
    data_name = ""
    df_preview = ""

    if data_file:
        dataset_uploaded = True
        data_name = data_file.filename or "upload"
        content = await data_file.read()
        suffix = os.path.splitext(data_name)[1] or ".bin"
        data_path = save_upload_to_temp(content, suffix)

        lower = data_name.lower()
        if lower.endswith((".png", ".jpg", ".jpeg", ".pdf")):
            text = ocr_space_extract_text(content, data_name)
            lines = [ln.rstrip() for ln in text.splitlines()]
            table_rows = lines_to_table_rows(lines)

            if table_rows:
                csv_path = write_rows_to_csv(table_rows)
                data_path = csv_path
                data_name = "ocr.csv"
            else:
                txt_path = save_upload_to_temp(text.encode("utf-8", errors="ignore"), ".txt")
                data_path = txt_path
                data_name = "ocr.txt"

        conn = duckdb.connect(database=":memory:")
        try:
            table, cols, rows = duckdb_load_table(conn, data_path, data_name)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Failed to load dataset into DuckDB: {e}")

        df_preview = (
            f"\n\nDataset loaded into DuckDB table `{table}`.\n"
            f"Columns: {', '.join(cols)}\n"
            f"Preview:\n{format_preview_md(cols, rows)}\n"
            "You can query it using `duckdb_conn` in generated code.\n"
            "Example: duckdb_conn.execute('SELECT COUNT(*) FROM data').fetchone()[0]\n"
        )

    rules: List[str] = []
    if dataset_uploaded:
        rules.append("1) You have access to a DuckDB connection `duckdb_conn` with a table named `data`.")
        rules.append("2) Do NOT fetch external data unless absolutely required.")
    else:
        rules.append("1) If you need web data, call `scrape_url_to_duckdb_table(url)` which loads it into table `data`.")
        rules.append("2) Do NOT use read_html(). DuckDB does not support read_html().")

    rules.append('3) Return ONLY a valid JSON object with keys: "keys" (list) and "code" (string).')
    rules.append("4) Your Python code MUST create a dict named `results` and fill it with exactly those keys.")
    rules.append("5) For charts, use `quickchart_to_base64(chart_config)` to return base64 PNG.")
    rules.append("6) Keep code deterministic; define variables before use; no prints other than building results.")
    rules.append("7) DO NOT import or use pandas, numpy, matplotlib, seaborn, pyarrow, or sklearn.")
    rules.append("8) Use DuckDB SQL ONLY for data manipulation.")
    rules.append("9) Use Python built-ins or DuckDB results (lists/tuples) for logic.")
    rules.append("10) When writing SQL inside Python strings: ALWAYS use raw strings r'...' OR double-escape backslashes (\\\\).")
    rules.append("11) Never embed tuples into SQL. `scrape_url_to_tempfile(url)` returns (path, filename) — prefer `scrape_url_to_duckdb_table(url)`.")

    prompt = (
        "You are a full-stack autonomous data analyst.\n\n"
        + "Rules:\n" + "\n".join(rules) + "\n\n"
        + "Questions (from .txt):\n" + raw_questions + "\n"
        + (df_preview if df_preview else "") + "\n"
        + 'Return ONLY JSON in this format:\n{"keys":["..."],"code":"..."}\n'
    )

    llm_out = call_llm(provider=provider, model=model, api_key=api_key, prompt=prompt)

    parsed = clean_llm_json(llm_out)
    if "error" in parsed:
        raise HTTPException(500, f"LLM output parsing error: {parsed.get('error')}. Raw: {parsed.get('raw')}")

    if not isinstance(parsed, dict) or "code" not in parsed or "keys" not in parsed:
        raise HTTPException(500, f"Invalid LLM response shape. Got: {parsed}")

    code = parsed["code"]
    keys = parsed["keys"]

    if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
        raise HTTPException(400, "LLM returned invalid `keys` (must be a list of strings).")

    exec_result = run_user_code(code=code, data_path=data_path, data_name=data_name, timeout=EXEC_TIMEOUT_SECONDS)
    if isinstance(exec_result, dict) and exec_result.get("status") == "error":
        raise HTTPException(500, f"Execution failed: {exec_result.get('message')}")

    final_out: Dict[str, Any] = {k: exec_result.get(k) for k in keys}
    return JSONResponse(content=final_out)

# ---------------------------------------------------------------------
# Favicon + health
# ---------------------------------------------------------------------
_FAVICON_FALLBACK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII="
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    path = "favicon.ico"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/x-icon")
    return Response(content=_FAVICON_FALLBACK_PNG, media_type="image/png")

@app.get("/api", include_in_schema=False)
async def api_info():
    return JSONResponse(
        {
            "ok": True,
            "message": "POST /api with questions_file (.txt), optional data_file, and provider/model/api_key.",
            "providers": list(PROVIDER_MODELS.keys()),
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
