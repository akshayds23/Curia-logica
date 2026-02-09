

import os
import re
import json
import base64
import tempfile
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import re
import json
import base64
import tempfile
import subprocess
import logging
from io import BytesIO
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi import FastAPI
from dotenv import load_dotenv

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io, os, json, re
import google.generativeai as genai

# Optional image conversion
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

from PIL import Image
# OCR (optional)
try:
    import pytesseract
    PYTESS_AVAILABLE = True
except Exception:
    pytesseract = None
    PYTESS_AVAILABLE = False

try:
    import cv2  # comes from opencv-python-headless
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    PYTESS_AVAILABLE = True
except ImportError:
    PYTESS_AVAILABLE = False

# NEW: PyMuPDF for PDFs
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

# LangChain / LLM imports (keep as you used)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
# NEW: Optional other providers via LangChain
try:
    from langchain_openai import ChatOpenAI
    HAS_LC_OPENAI = True
except Exception:
    HAS_LC_OPENAI = False

try:
    from langchain_anthropic import ChatAnthropic
    HAS_LC_ANTHROPIC = True
except Exception:
    HAS_LC_ANTHROPIC = False

from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TDS Data Analyst Agent")

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 180))


from pathlib import Path

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    base_dir = Path(__file__).resolve().parent
    html_path = base_dir / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            content="<h1>Frontend not found</h1><p>index.html is missing.</p>",
            status_code=404
        )
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))




# -----------------------------
# Tools
# -----------------------------

@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """
    Universal web/data scraper.
    Fetches data from any URL: JSON, CSV, Excel, Parquet, DB files, archives, HTML tables, or dynamic JS-rendered pages.
    Returns a dictionary with status, data, and columns.
    """
    import os, re, tempfile, requests, pandas as pd, duckdb
    from io import BytesIO, StringIO
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.google.com"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()

        # JSON
        if "application/json" in ctype or url.endswith(".json"):
            df = pd.json_normalize(resp.json())
            return {"status": "success", "data": df.to_dict(orient="records"), "columns": list(df.columns)}

        # CSV
        if "text/csv" in ctype or url.endswith(".csv"):
            df = pd.read_csv(BytesIO(resp.content))
            return {"status": "success", "data": df.to_dict(orient="records"), "columns": list(df.columns)}

        # Excel
        if any(url.endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in ctype:
            df = pd.read_excel(BytesIO(resp.content))
            return {"status": "success", "data": df.to_dict(orient="records"), "columns": list(df.columns)}

        # Parquet
        if url.endswith(".parquet") or "parquet" in ctype:
            df = pd.read_parquet(BytesIO(resp.content))
            return {"status": "success", "data": df.to_dict(orient="records"), "columns": list(df.columns)}

        # Databases (.db, .duckdb)
        if url.endswith(".db") or url.endswith(".duckdb"):
            tmp_path = tempfile.NamedTemporaryFile(delete=False).name
            with open(tmp_path, "wb") as f:
                f.write(resp.content)
            con = duckdb.connect(database=':memory:')
            con.execute(f"ATTACH '{tmp_path}' AS db")
            tables = con.execute("SHOW TABLES FROM db").fetchdf()
            if not tables.empty:
                table_name = tables.iloc[0, 0]
                df = con.execute(f"SELECT * FROM db.{table_name}").fetchdf()
                con.close()
                os.remove(tmp_path)
                return {"status": "success", "data": df.to_dict(orient="records"), "columns": list(df.columns)}

        # Archives (.tar.gz, .zip)
        if url.endswith((".tar.gz", ".tgz", ".tar", ".zip")):
            import tarfile, zipfile
            content = BytesIO(resp.content)
            if url.endswith(".zip"):
                with zipfile.ZipFile(content, 'r') as z:
                    for name in z.namelist():
                        if name.endswith(".parquet"):
                            df = pd.read_parquet(z.open(name))
                            return {"status": "success", "data": df.to_dict(orient="records"), "columns": list(df.columns)}
                        if name.endswith(".csv"):
                            df = pd.read_csv(z.open(name))
                            return {"status": "success", "data": df.to_dict(orient="records"), "columns": list(df.columns)}
            else:
                with tarfile.open(fileobj=content, mode="r:*") as tar:
                    for member in tar.getmembers():
                        if member.name.endswith(".parquet"):
                            df = pd.read_parquet(tar.extractfile(member))
                            return {"status": "success", "data": df.to_dict(orient="records"), "columns": list(df.columns)}
                        if member.name.endswith(".csv"):
                            df = pd.read_csv(tar.extractfile(member))
                            return {"status": "success", "data": df.to_dict(orient="records"), "columns": list(df.columns)}

        # Static HTML tables
        try:
            tables = pd.read_html(StringIO(resp.text), flavor="lxml")
            if tables:
                df = tables[0]
                return {"status": "success", "data": df.to_dict(orient="records"), "columns": list(df.columns)}
        except Exception:
            pass

        # Dynamic JS rendering
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=45000)
                page.wait_for_load_state("networkidle")
                html = page.content()
                browser.close()
            tables = pd.read_html(StringIO(html), flavor="lxml")
            if tables:
                df = tables[0]
                return {"status": "success", "data": df.to_dict(orient="records"), "columns": list(df.columns)}
        except Exception:
            pass

        # Plain text fallback
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "lxml")
        text = soup.get_text("\n", strip=True)
        return {"status": "success", "data": [{"text": text}], "columns": ["text"]}

    except Exception as e:
        return {"status": "error", "message": str(e)}



# -----------------------------
# Utilities for executing code safely
# -----------------------------
def clean_llm_output(output: str) -> Dict:
    """
    Extract JSON object from LLM output robustly.
    Returns dict or {"error": "…"}
    """
    try:
        if not output:
            return {"error": "Empty LLM output"}
        # remove triple-fence markers if present
        s = re.sub(r"^```(?:json)?\s*", "", output.strip())
        s = re.sub(r"\s*```$", "", s)
        # find outermost JSON object by scanning for balanced braces
        first = s.find("{")
        last = s.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return {"error": "No JSON object found in LLM output", "raw": s}
        candidate = s[first:last+1]
        try:
            return json.loads(candidate)
        except Exception as e:
            # fallback: try last balanced pair scanning backwards
            for i in range(last, first, -1):
                cand = s[first:i+1]
                try:
                    return json.loads(cand)
                except Exception:
                    continue
            return {"error": f"JSON parsing failed: {str(e)}", "raw": candidate}
    except Exception as e:
        return {"error": str(e)}

SCRAPE_FUNC = r'''
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5
        )
        response.raise_for_status()
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": [],
            "columns": []
        }

    soup = BeautifulSoup(response.text, "html.parser")
    tables = pd.read_html(response.text)

    if tables:
        df = tables[0]  # Take first table
        df.columns = [str(c).strip() for c in df.columns]
        
        # Ensure all columns are unique and string
        df.columns = [str(col) for col in df.columns]

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
    else:
        # Fallback to plain text
        text_data = soup.get_text(separator="\n", strip=True)

        # Try to detect possible "keys" from text like Runtime, Genre, etc.
        detected_cols = set(re.findall(r"\b[A-Z][a-zA-Z ]{2,15}\b", text_data))
        df = pd.DataFrame([{}])  # start empty
        for col in detected_cols:
            df[col] = None

        if df.empty:
            df["text"] = [text_data]

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
'''


def write_and_run_temp_python(code: str, injected_pickle: str = None, timeout: int = 60) -> Dict[str, Any]:
    """
    Write a temp python file which:
      - provides a safe environment (imports)
      - loads df/from pickle if provided into df and data variables
      - defines a robust plot_to_base64() helper that ensures < 100kB (attempts resizing/conversion)
      - executes the user code (which should populate `results` dict)
      - prints json.dumps({"status":"success","result":results})
    Returns dict with parsed JSON or error details.
    """
    # create file content
    preamble = [
        "import json, sys, gc",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from io import BytesIO",
        "import base64",
    ]
    if PIL_AVAILABLE:
        preamble.append("from PIL import Image")
    # inject df if a pickle path provided
    if injected_pickle:
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')\n")
        preamble.append("data = df.to_dict(orient='records')\n")
    else:
        # ensure data exists so user code that references data won't break
        preamble.append("data = globals().get('data', {})\n")

    # plot_to_base64 helper that tries to reduce size under 100_000 bytes
    helper = r'''
def plot_to_base64(max_bytes=100000):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_bytes = buf.getvalue()
    if len(img_bytes) <= max_bytes:
        return base64.b64encode(img_bytes).decode('ascii')
    # try decreasing dpi/figure size iteratively
    for dpi in [80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return base64.b64encode(b).decode('ascii')
    # if Pillow available, try convert to WEBP which is typically smaller
    try:
        from PIL import Image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=40)
        buf.seek(0)
        im = Image.open(buf)
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=80, method=6)
        out_buf.seek(0)
        ob = out_buf.getvalue()
        if len(ob) <= max_bytes:
            return base64.b64encode(ob).decode('ascii')
        # try lower quality
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=60, method=6)
        out_buf.seek(0)
        ob = out_buf.getvalue()
        if len(ob) <= max_bytes:
            return base64.b64encode(ob).decode('ascii')
    except Exception:
        pass
    # as last resort return downsized PNG even if > max_bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')
'''

    # Build the code to write
    script_lines = []
    script_lines.extend(preamble)
    script_lines.append(helper)
    script_lines.append(SCRAPE_FUNC)
    script_lines.append("\nresults = {}\n")
    script_lines.append(code)
    # ensure results printed as json
    script_lines.append("\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)\n")

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
    tmp.write("\n".join(script_lines))
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    try:
        completed = subprocess.run([sys.executable, tmp_path],
                                   capture_output=True, text=True, timeout=timeout)
        if completed.returncode != 0:
            # collect stderr and stdout for debugging
            return {"status": "error", "message": completed.stderr.strip() or completed.stdout.strip()}
        # parse stdout as json
        out = completed.stdout.strip()
        try:
            parsed = json.loads(out)
            return parsed
        except Exception as e:
            return {"status": "error", "message": f"Could not parse JSON output: {str(e)}", "raw": out}
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out"}
    finally:
        try:
            os.unlink(tmp_path)
            if injected_pickle and os.path.exists(injected_pickle):
                os.unlink(injected_pickle)
        except Exception:
            pass



# LLM agent setup (safe, lazy default)
# -----------------------------

_GAPI = os.getenv("GOOGLE_API_KEY", "")

llm = None
agent = None
agent_executor = None

if _GAPI:
    # Only build default Gemini if a key is present to avoid ADC fallback
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.5-pro"),
        temperature=0,
        api_key=_GAPI,  # IMPORTANT: use api_key (not google_api_key)
    )

    # Tools list for agent (LangChain tool decorator returns metadata for the LLM)
    tools = [scrape_url_to_dataframe]  # we only expose scraping as a tool; agent will still produce code

    # Prompt: instruct agent to call the tool and output JSON only
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a full-stack autonomous data analyst agent.

You will receive:
- A set of **rules** for this request
- One or more **questions**
- An optional **dataset preview**
- A `.txt` file that specifies the required JSON keys and their types.

You must:
1. Follow the provided rules exactly.
2. Return only a valid JSON object — no extra commentary or formatting.
3. The JSON must contain:
   - "keys": [ list of output keys exactly as specified in the .txt file ]
   - "code": "..." (Python code that creates a dict called `results` with each output key as a key and its computed answer as the value)
4. In your Python code, make sure the values are cast to the types specified in the .txt file:
   - `number` → float
   - `integer` / `int` → int
   - `string` → str
   - `bar_chart` / `plt` etc. → base64 PNG string under 100kB (use plot_to_base64()).
5. Do not return the full question text as a key. Always use the JSON key specified in the `.txt`.
6. Always define variables before use. Code must run without errors.
"""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(
        llm=llm,
        tools=[scrape_url_to_dataframe],
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=[scrape_url_to_dataframe],
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        return_intermediate_steps=False
    )
else:
    # Build prompt anyway so per-request agents can use it
    tools = [scrape_url_to_dataframe]
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a full-stack autonomous data analyst agent.

You will receive:
- A set of **rules** for this request
- One or more **questions**
- An optional **dataset preview**
- A `.txt` file that specifies the required JSON keys and their types.

You must:
1. Follow the provided rules exactly.
2. Return only a valid JSON object — no extra commentary or formatting.
3. The JSON must contain:
   - "keys": [ list of output keys exactly as specified in the .txt file ]
   - "code": "..." (Python code that creates a dict called `results` with each output key as a key and its computed answer as the value)
4. In your Python code, make sure the values are cast to the types specified in the .txt file:
   - `number` → float
   - `integer` / `int` → int
   - `string` → str
   - `bar_chart` / `plt` etc. → base64 PNG string under 100kB (use plot_to_base64()).
5. Do not return the full question text as a key. Always use the JSON key specified in the `.txt`.
6. Always define variables before use. Code must run without errors.
"""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


def make_llm(provider: str, model: str = "", api_key: str = ""):
    """
    Build a LangChain chat model for the chosen provider.
    Falls back to env vars if api_key is empty.
    """
    provider = (provider or "gemini").lower()
    if provider == "openai":
        if not HAS_LC_OPENAI:
            raise HTTPException(500, "OpenAI provider not installed (langchain-openai).")
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=0,
            api_key=api_key or os.getenv("OPENAI_API_KEY", "")
        )

    if provider == "claude":
        if not HAS_LC_ANTHROPIC:
            raise HTTPException(500, "Claude provider not installed (langchain-anthropic).")
        return ChatAnthropic(
            model=model or "claude-3-5-sonnet-20241022",
            temperature=0,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY", "")
        )

    # default: Gemini
    gkey = api_key or os.getenv("GOOGLE_API_KEY", "")
    if not gkey:
        raise HTTPException(500, "GOOGLE_API_KEY is required for Gemini (pass api_key or set env).")
    return ChatGoogleGenerativeAI(
        model=model or os.getenv("GOOGLE_MODEL", "gemini-2.5-pro"),
        temperature=0,
        api_key=gkey,  # IMPORTANT
    )


def build_agent_for(selected_llm):
    """Create an AgentExecutor using your same prompt/tools."""
    a = create_tool_calling_agent(
        llm=selected_llm,
        tools=[scrape_url_to_dataframe],
        prompt=prompt
    )
    return AgentExecutor(
        agent=a,
        tools=[scrape_url_to_dataframe],
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        return_intermediate_steps=False
    )

# -----------------------------
# OCR helpers (images) and PDF via PyMuPDF
# -----------------------------
def _lines_to_table_df(lines: List[str]) -> pd.DataFrame | None:
    """
    Simple heuristic: split on 2+ spaces, keep rows with the modal column count.
    """
    rows = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        parts = re.split(r"\s{2,}", s)
        rows.append(parts)
    if not rows:
        return None
    counts = {}
    for r in rows:
        counts[len(r)] = counts.get(len(r), 0) + 1
    best_cols = max(counts, key=counts.get)
    if best_cols < 2:
        return None
    filtered = [r for r in rows if len(r) == best_cols]
    if len(filtered) < max(3, int(0.4 * len(rows))):
        return None
    header = None
    maybe_header = filtered[0]
    if len(set(maybe_header)) == len(maybe_header):
        header = [str(c).strip() for c in maybe_header]
    if header:
        df = pd.DataFrame(filtered[1:], columns=header)
    else:
        df = pd.DataFrame(filtered)
    df = df.apply(pd.to_numeric, errors="ignore")
    return df

def ocr_image_to_df(image_bytes: bytes) -> pd.DataFrame:
    if not (PIL_AVAILABLE and PYTESS_AVAILABLE):
        raise HTTPException(400, "OCR requires Pillow + pytesseract installed.")
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    text = pytesseract.image_to_string(image)
    lines = [ln.rstrip() for ln in text.splitlines()]
    df_tab = _lines_to_table_df(lines)
    if df_tab is not None and not df_tab.empty:
        return df_tab
    return pd.DataFrame({"ocr_text": [text]})

def pdf_to_dataframe_with_pymupdf(pdf_bytes: bytes) -> pd.DataFrame:
    if not HAS_PYMUPDF:
        raise HTTPException(400, "PyMuPDF (fitz) is required for PDF handling.")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_texts: List[str] = [p.get_text("text") for p in doc]
    # OCR pass (render pages and OCR) if possible
    ocr_lines: List[str] = []
    if PIL_AVAILABLE and PYTESS_AVAILABLE:
        for p in doc:
            mat = fitz.Matrix(2, 2)
            pix = p.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
            try:
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                text = pytesseract.image_to_string(image)
                ocr_lines.extend([ln.rstrip() for ln in text.splitlines()])
            except Exception:
                continue
    # try to reconstruct table first
    if ocr_lines:
        df = _lines_to_table_df(ocr_lines)
        if df is not None and not df.empty:
            return df
    # fallback to text per page
    if any(s.strip() for s in page_texts):
        return pd.DataFrame({"page": list(range(1, len(page_texts)+1)), "text": page_texts})
    # last resort: one big OCR blob
    if ocr_lines:
        return pd.DataFrame({"text": ["\n".join(ocr_lines)]})
    return pd.DataFrame({"note": ["No extractable content"]})

# -----------------------------
# Runner: orchestrates agent -> pre-scrape inject -> execute
# -----------------------------

from fastapi import Request

@app.post("/api")
async def analyze_data(request: Request):
    try:
        form = await request.form()
        questions_file = None
        data_file = None

        # NEW: provider fields from UI (non-breaking; defaults to Gemini)
        provider = (form.get("provider") or "gemini").strip()
        model    = (form.get("model") or "").strip()
        api_key  = (form.get("api_key") or "").strip()

        for key, val in form.items():
            if hasattr(val, "filename") and val.filename:  # it's a file
                fname = val.filename.lower()
                if fname.endswith(".txt") and questions_file is None:
                    questions_file = val
                else:
                    data_file = val

        if not questions_file:
            raise HTTPException(400, "Missing questions file (.txt)")

        raw_questions = (await questions_file.read()).decode("utf-8")
        
        pickle_path = None
        df_preview = ""
        dataset_uploaded = False

        if data_file:
            dataset_uploaded = True
            filename = data_file.filename.lower()
            content = await data_file.read()
            from io import BytesIO
            import duckdb, tempfile, tarfile, zipfile

            df = None
            duckdb_conn = duckdb.connect(database=':memory:')

            # CSV
            if filename.endswith(".csv"):
                df = pd.read_csv(BytesIO(content))
                duckdb_conn.register("df", df)

            # Excel
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(BytesIO(content))
                duckdb_conn.register("df", df)
            
            # NEW: PDF via PyMuPDF (no LLM)
            elif filename.lower().endswith(".pdf"):
                df = pdf_to_dataframe_with_pymupdf(content)
                duckdb_conn.register("df", df)

            # Parquet
            elif filename.endswith(".parquet"):
                df = pd.read_parquet(BytesIO(content))
                duckdb_conn.register("df", df)

            # SQLite / DuckDB database
            elif filename.endswith(".db") or filename.endswith(".duckdb"):
                tmp_path = tempfile.NamedTemporaryFile(delete=False).name
                with open(tmp_path, "wb") as f:
                    f.write(content)
                duckdb_conn.execute(f"ATTACH '{tmp_path}' AS uploaded_db")
                # Pick the first table for df
                tables = duckdb_conn.execute("SHOW TABLES FROM uploaded_db").fetchdf()
                if not tables.empty:
                    first_table = tables.iloc[0, 0]
                    df = duckdb_conn.execute(f"SELECT * FROM uploaded_db.{first_table}").fetchdf()

            # Archives (.tar.gz, .zip)
            elif filename.endswith((".tar.gz", ".tgz", ".tar", ".zip")):
                content_io = BytesIO(content)
                if filename.endswith(".zip"):
                    with zipfile.ZipFile(content_io, 'r') as z:
                        for name in z.namelist():
                            if name.endswith(".parquet"):
                                df = pd.read_parquet(z.open(name))
                                break
                            if name.endswith(".csv"):
                                df = pd.read_csv(z.open(name))
                                break
                else:
                    with tarfile.open(fileobj=content_io, mode="r:*") as tar:
                        for member in tar.getmembers():
                            if member.name.endswith(".parquet"):
                                df = pd.read_parquet(tar.extractfile(member))
                                break
                            if member.name.endswith(".csv"):
                                df = pd.read_csv(tar.extractfile(member))
                                break
                if df is not None:
                    duckdb_conn.register("df", df)

            # JSON
            elif filename.endswith(".json"):
                try:
                    df = pd.read_json(BytesIO(content))
                except ValueError:
                    df = pd.DataFrame(json.loads(content.decode("utf-8")))
                duckdb_conn.register("df", df)

            # NEW: Images via OCR
            elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
                df = ocr_image_to_df(content)
                duckdb_conn.register("df", df)

            else:
                raise HTTPException(400, f"Unsupported data file type: {filename}")

            # Save pickle for LLM code injection
            temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            temp_pkl.close()
            df.to_pickle(temp_pkl.name)
            pickle_path = temp_pkl.name

            # Inject duckdb_conn into execution environment
            df_preview = (
                f"\n\nThe uploaded dataset has {len(df)} rows and {len(df.columns)} columns.\n"
                f"Columns: {', '.join(df.columns.astype(str))}\n"
                f"First rows:\n{df.head(5).to_markdown(index=False)}\n"
                f"You can also query the dataset using DuckDB via the variable `duckdb_conn`.\n"
            )

        # Build rules based on data presence
        if dataset_uploaded:
            llm_rules = (
                "Rules:\n"
                "1) You have access to a pandas DataFrame called `df` and its dictionary form `data`.\n"
                "2) DO NOT call scrape_url_to_dataframe() or fetch any external data.\n"
            )
        else:
            llm_rules = (
                "Rules:\n"
                "1) If you need web data, CALL scrape_url_to_dataframe(url).\n"
            )

        llm_input = (
            f"{llm_rules}\nQuestions:\n{raw_questions}\n"
            f"{df_preview if df_preview else ''}"
            "Respond with the JSON object only."
        )

        # NEW: Build per-request LLM + Agent (keeps your global Gemini default)
        try:
            selected_llm = make_llm(provider=provider, model=model, api_key=api_key)
        except HTTPException as e:
            # fall back to default Gemini if provider packages not installed
            selected_llm = llm
        agent_override = build_agent_for(selected_llm)

        # Run agent
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            fut = ex.submit(run_agent_safely_unified, llm_input, pickle_path, agent_override)
            try:
                result = fut.result(timeout=LLM_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                raise HTTPException(408, "Processing timeout")

        if "error" in result:
            raise HTTPException(500, detail=result["error"])
        print(result)
        return JSONResponse(content=result)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("analyze_data failed")
        raise HTTPException(500, detail=str(e))


def run_agent_safely_unified(llm_input: str, pickle_path: str = None, agent_override=None) -> Dict:
    """
    Runs the LLM agent and executes code.
    - Retries up to 3 times if agent returns no output.
    - If pickle_path is provided, injects that DataFrame directly.
    - If no pickle_path, falls back to scraping when needed.
    """
    try:
        max_retries = 5
        raw_out = ""
        # choose which agent to use
        agent_to_use = agent_override or agent_executor
        for attempt in range(1, max_retries + 1):
            response = agent_to_use.invoke({"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS})
            raw_out = response.get("output") or response.get("final_output") or response.get("text") or ""
            if raw_out:
                break
        if not raw_out:
            return {"error": f"Agent returned no output after {max_retries} attempts"}

        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            return parsed

        if not isinstance(parsed, dict) or "code" not in parsed or ("questions" not in parsed and "keys" not in parsed):
            return {"error": f"Invalid agent response format: {parsed}"}

        code = parsed["code"]
        if pickle_path is None:
            urls = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", code)
            if urls:
                url = urls[0]
                tool_resp = scrape_url_to_dataframe(url)
                if tool_resp.get("status") != "success":
                    return {"error": f"Scrape tool failed: {tool_resp.get('message')}"}
                df = pd.DataFrame(tool_resp["data"])
                temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                temp_pkl.close()
                df.to_pickle(temp_pkl.name)
                pickle_path = temp_pkl.name

        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
        if exec_result.get("status") != "success":
            return {"error": f"Execution failed: {exec_result.get('message')}", "raw": exec_result.get("raw")}

        results_dict = exec_result.get("result", {})
        return results_dict

    except Exception as e:
        logger.exception("run_agent_safely_unified failed")
        return {"error": str(e)}


    
from fastapi.responses import FileResponse, Response
import base64, os

# 1×1 transparent PNG fallback (if favicon.ico file not present)
_FAVICON_FALLBACK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII="
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Serve favicon.ico if present in the working directory.
    Otherwise return a tiny transparent PNG to avoid 404s.
    """
    path = "favicon.ico"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/x-icon")
    return Response(content=_FAVICON_FALLBACK_PNG, media_type="image/png")

@app.get("/api", include_in_schema=False)
async def analyze_get_info():
    """Health/info endpoint. Use POST /api for actual analysis."""
    return JSONResponse({
        "ok": True,
        "message": "Server is running. Use POST /api with 'questions_file' and optional 'data_file'.",
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
