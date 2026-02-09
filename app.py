import os
import re
import json
import base64
import tempfile
import subprocess
import logging
import sys
import io
from io import BytesIO
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
from dotenv import load_dotenv

# LangChain / LLM imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Optional other providers via LangChain
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

# Optional image support
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# Optional PDF support (PyMuPDF)
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TDS Data Analyst Agent")

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 180))

# OCR.space config (required for OCR)
OCRSPACE_API_KEY = os.getenv("OCRSPACE_API_KEY", "")
OCRSPACE_LANGUAGE = os.getenv("OCRSPACE_LANGUAGE", "eng")
OCRSPACE_TIMEOUT = int(os.getenv("OCRSPACE_TIMEOUT_SECONDS", 60))
OCRSPACE_MAX_PDF_PAGES = int(os.getenv("OCRSPACE_MAX_PDF_PAGES", 3))  # keep small for serverless


# -----------------------------
# Frontend route
# -----------------------------
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
# OCR.space helpers
# -----------------------------
def ocr_image_with_ocrspace(image_bytes: bytes, language: str = "eng") -> str:
    if not OCRSPACE_API_KEY:
        raise HTTPException(400, "OCRSPACE_API_KEY is not set (required for OCR).")

    try:
        resp = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": ("image.png", image_bytes)},
            data={
                "apikey": OCRSPACE_API_KEY,
                "language": language,
                "isOverlayRequired": False,
            },
            timeout=OCRSPACE_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
    except requests.RequestException as e:
        raise HTTPException(502, f"OCR.space request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(502, f"OCR.space response parse failed: {str(e)}")

    if result.get("IsErroredOnProcessing"):
        msg = result.get("ErrorMessage") or result.get("ErrorDetails") or "Unknown OCR error"
        raise HTTPException(502, f"OCR.space error: {msg}")

    parsed = result.get("ParsedResults") or []
    if not parsed:
        return ""
    return (parsed[0].get("ParsedText") or "").strip()


def _lines_to_table_df(lines: List[str]) -> Optional[pd.DataFrame]:
    """
    Simple heuristic: split on 2+ spaces, keep rows with the modal column count.
    """
    rows: List[List[str]] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        parts = re.split(r"\s{2,}", s)
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
    # OCR.space accepts bytes directly; no need for PIL
    text = ocr_image_with_ocrspace(image_bytes, language=OCRSPACE_LANGUAGE)
    lines = [ln.rstrip() for ln in text.splitlines()]
    df_tab = _lines_to_table_df(lines)
    if df_tab is not None and not df_tab.empty:
        return df_tab
    return pd.DataFrame({"ocr_text": [text]})


def pdf_to_dataframe_with_pymupdf(pdf_bytes: bytes) -> pd.DataFrame:
    if not HAS_PYMUPDF:
        raise HTTPException(400, "PyMuPDF (fitz) is required for PDF handling.")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # 1) Try normal text extraction first
    page_texts: List[str] = [p.get_text("text") for p in doc]
    if any(t.strip() for t in page_texts):
        return pd.DataFrame({"page": list(range(1, len(page_texts) + 1)), "text": page_texts})

    # 2) If scanned / no text, render a few pages and OCR via OCR.space
    ocr_lines: List[str] = []
    pages_to_ocr = min(len(doc), max(1, OCRSPACE_MAX_PDF_PAGES))
    for i in range(pages_to_ocr):
        p = doc[i]
        mat = fitz.Matrix(2, 2)
        pix = p.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        try:
            text = ocr_image_with_ocrspace(img_bytes, language=OCRSPACE_LANGUAGE)
            ocr_lines.extend([ln.rstrip() for ln in text.splitlines()])
        except HTTPException:
            # keep going; we'll fallback later if all fail
            continue
        except Exception:
            continue

    if ocr_lines:
        df = _lines_to_table_df(ocr_lines)
        if df is not None and not df.empty:
            return df
        return pd.DataFrame({"text": ["\n".join(ocr_lines)]})

    return pd.DataFrame({"note": ["No extractable content (text extraction empty; OCR failed or disabled)."]})


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
    import duckdb
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
            con = duckdb.connect(database=":memory:")
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
                with zipfile.ZipFile(content, "r") as z:
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

        # Plain text fallback
        soup = BeautifulSoup(resp.text, "lxml")
        text = soup.get_text("\n", strip=True)
        return {"status": "success", "data": [{"text": text}], "columns": ["text"]}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# -----------------------------
# Utilities for executing code safely
# -----------------------------
def clean_llm_output(output: str) -> Dict[str, Any]:
    """
    Extract JSON object from LLM output robustly.
    Returns dict or {"error": "..."}
    """
    try:
        if not output:
            return {"error": "Empty LLM output"}
        s = re.sub(r"^```(?:json)?\s*", "", output.strip())
        s = re.sub(r"\s*```$", "", s)
        first = s.find("{")
        last = s.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return {"error": "No JSON object found in LLM output", "raw": s}
        candidate = s[first:last + 1]
        try:
            return json.loads(candidate)
        except Exception as e:
            for i in range(last, first, -1):
                cand = s[first:i + 1]
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
from io import StringIO

def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
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
    try:
        tables = pd.read_html(StringIO(response.text))
    except Exception:
        tables = []

    if tables:
        df = tables[0]
        df.columns = [str(c).strip() for c in df.columns]
        df.columns = [str(col) for col in df.columns]
        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
    else:
        text_data = soup.get_text(separator="\n", strip=True)
        detected_cols = set(re.findall(r"\b[A-Z][a-zA-Z ]{2,15}\b", text_data))
        df = pd.DataFrame([{}])
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


def write_and_run_temp_python(code: str, injected_pickle: Optional[str] = None, timeout: int = 60) -> Dict[str, Any]:
    """
    Write a temp python file that loads df/from pickle if provided, defines plot_to_base64(),
    defines scrape_url_to_dataframe() (light version), then executes code that populates `results`.
    """
    preamble = [
        "import json, sys, gc",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from io import BytesIO",
        "import base64",
    ]

    if injected_pickle:
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')\n")
        preamble.append("data = df.to_dict(orient='records')\n")
    else:
        preamble.append("data = globals().get('data', {})\n")

    helper = r'''
def plot_to_base64(max_bytes=100000):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_bytes = buf.getvalue()
    if len(img_bytes) <= max_bytes:
        return base64.b64encode(img_bytes).decode('ascii')
    for dpi in [80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return base64.b64encode(b).decode('ascii')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')
'''

    script_lines: List[str] = []
    script_lines.extend(preamble)
    script_lines.append(helper)
    script_lines.append(SCRAPE_FUNC)
    script_lines.append("\nresults = {}\n")
    script_lines.append(code)
    script_lines.append("\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)\n")

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    tmp.write("\n".join(script_lines))
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    try:
        completed = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=timeout)
        if completed.returncode != 0:
            return {"status": "error", "message": (completed.stderr.strip() or completed.stdout.strip())}
        out = completed.stdout.strip()
        try:
            return json.loads(out)
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


# -----------------------------
# LLM agent setup (safe, lazy default)
# -----------------------------
_GAPI = os.getenv("GOOGLE_API_KEY", "")

llm = None
agent_executor = None

tools = [scrape_url_to_dataframe]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a full-stack autonomous data analyst agent.

You will receive:
- A set of rules for this request
- One or more questions
- An optional dataset preview
- A .txt file that specifies the required JSON keys and their types.

You must:
1. Follow the provided rules exactly.
2. Return only a valid JSON object — no extra commentary or formatting.
3. The JSON must contain:
   - "keys": [ list of output keys exactly as specified in the .txt file ]
   - "code": "..." (Python code that creates a dict called `results` with each output key as a key and its computed answer as the value)
4. In your Python code, make sure the values are cast to the types specified in the .txt file:
   - number -> float
   - integer/int -> int
   - string -> str
   - bar_chart/plt etc. -> base64 PNG string under 100kB (use plot_to_base64()).
5. Do not return the full question text as a key. Always use the JSON key specified in the .txt.
6. Always define variables before use. Code must run without errors.
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

if _GAPI:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.5-pro"),
        temperature=0,
        api_key=_GAPI,
    )
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )


def make_llm(provider: str, model: str = "", api_key: str = ""):
    provider = (provider or "gemini").lower()

    if provider == "openai":
        if not HAS_LC_OPENAI:
            raise HTTPException(500, "OpenAI provider not installed (langchain-openai).")
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise HTTPException(500, "OPENAI_API_KEY is required for OpenAI provider.")
        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=0, api_key=key)

    if provider == "claude":
        if not HAS_LC_ANTHROPIC:
            raise HTTPException(500, "Claude provider not installed (langchain-anthropic).")
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise HTTPException(500, "ANTHROPIC_API_KEY is required for Claude provider.")
        return ChatAnthropic(model=model or "claude-3-5-sonnet-20241022", temperature=0, api_key=key)

    # default: Gemini
    gkey = api_key or os.getenv("GOOGLE_API_KEY", "")
    if not gkey:
        raise HTTPException(500, "GOOGLE_API_KEY is required for Gemini provider.")
    return ChatGoogleGenerativeAI(
        model=model or os.getenv("GOOGLE_MODEL", "gemini-2.5-pro"),
        temperature=0,
        api_key=gkey,
    )


def build_agent_for(selected_llm):
    a = create_tool_calling_agent(llm=selected_llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=a,
        tools=tools,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )


# -----------------------------
# Runner: orchestrates agent -> pre-scrape inject -> execute
# -----------------------------
@app.post("/api")
async def analyze_data(request: Request):
    try:
        form = await request.form()
        questions_file = None
        data_file = None

        provider = (form.get("provider") or "gemini").strip()
        model = (form.get("model") or "").strip()
        api_key = (form.get("api_key") or "").strip()

        for _, val in form.items():
            if hasattr(val, "filename") and val.filename:
                fname = val.filename.lower()
                if fname.endswith(".txt") and questions_file is None:
                    questions_file = val
                else:
                    data_file = val

        if not questions_file:
            raise HTTPException(400, "Missing questions file (.txt)")

        raw_questions = (await questions_file.read()).decode("utf-8")

        pickle_path: Optional[str] = None
        df_preview = ""
        dataset_uploaded = False

        if data_file:
            dataset_uploaded = True
            filename = data_file.filename.lower()
            content = await data_file.read()
            from io import BytesIO
            import duckdb
            import tarfile
            import zipfile

            df: Optional[pd.DataFrame] = None
            duckdb_conn = duckdb.connect(database=":memory:")

            if filename.endswith(".csv"):
                df = pd.read_csv(BytesIO(content))
                duckdb_conn.register("df", df)

            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(BytesIO(content))
                duckdb_conn.register("df", df)

            elif filename.endswith(".pdf"):
                df = pdf_to_dataframe_with_pymupdf(content)
                duckdb_conn.register("df", df)

            elif filename.endswith(".parquet"):
                df = pd.read_parquet(BytesIO(content))
                duckdb_conn.register("df", df)

            elif filename.endswith(".db") or filename.endswith(".duckdb"):
                tmp_path = tempfile.NamedTemporaryFile(delete=False).name
                with open(tmp_path, "wb") as f:
                    f.write(content)
                duckdb_conn.execute(f"ATTACH '{tmp_path}' AS uploaded_db")
                tables = duckdb_conn.execute("SHOW TABLES FROM uploaded_db").fetchdf()
                if not tables.empty:
                    first_table = tables.iloc[0, 0]
                    df = duckdb_conn.execute(f"SELECT * FROM uploaded_db.{first_table}").fetchdf()

            elif filename.endswith((".tar.gz", ".tgz", ".tar", ".zip")):
                content_io = BytesIO(content)
                if filename.endswith(".zip"):
                    with zipfile.ZipFile(content_io, "r") as z:
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

            elif filename.endswith(".json"):
                try:
                    df = pd.read_json(BytesIO(content))
                except ValueError:
                    df = pd.DataFrame(json.loads(content.decode("utf-8")))
                duckdb_conn.register("df", df)

            elif filename.endswith((".png", ".jpg", ".jpeg")):
                df = ocr_image_to_df(content)
                duckdb_conn.register("df", df)

            else:
                raise HTTPException(400, f"Unsupported data file type: {filename}")

            if df is None:
                raise HTTPException(400, "Could not load dataset from uploaded file.")

            temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            temp_pkl.close()
            df.to_pickle(temp_pkl.name)
            pickle_path = temp_pkl.name

            df_preview = (
                f"\n\nThe uploaded dataset has {len(df)} rows and {len(df.columns)} columns.\n"
                f"Columns: {', '.join(df.columns.astype(str))}\n"
                f"First rows:\n{df.head(5).to_markdown(index=False)}\n"
                f"You can also query the dataset using DuckDB via the variable `duckdb_conn`.\n"
            )

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

        # Build per-request LLM + Agent (fallback to default gemini if available)
        try:
            selected_llm = make_llm(provider=provider, model=model, api_key=api_key)
            agent_override = build_agent_for(selected_llm)
        except HTTPException:
            if not agent_executor:
                raise HTTPException(500, "No default Gemini agent available. Set GOOGLE_API_KEY or provide provider api_key.")
            agent_override = agent_executor

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            fut = ex.submit(run_agent_safely_unified, llm_input, pickle_path, agent_override)
            try:
                result = fut.result(timeout=LLM_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                raise HTTPException(408, "Processing timeout")

        if "error" in result:
            raise HTTPException(500, detail=result["error"])

        return JSONResponse(content=result)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("analyze_data failed")
        raise HTTPException(500, detail=str(e))


def run_agent_safely_unified(llm_input: str, pickle_path: Optional[str] = None, agent_to_use=None) -> Dict[str, Any]:
    """
    Runs the LLM agent and executes code.
    - Retries if agent returns no output.
    - If pickle_path is provided, injects that DataFrame directly.
    - If no pickle_path, allows scraping tool injection if code calls it.
    """
    try:
        max_retries = 5
        raw_out = ""

        if agent_to_use is None:
            if agent_executor is None:
                return {"error": "No agent configured (missing GOOGLE_API_KEY and no provider api_key provided)."}
            agent_to_use = agent_executor

        for _ in range(max_retries):
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

        return exec_result.get("result", {})

    except Exception as e:
        logger.exception("run_agent_safely_unified failed")
        return {"error": str(e)}


# -----------------------------
# Favicon + health
# -----------------------------
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
async def analyze_get_info():
    return JSONResponse({
        "ok": True,
        "message": "Server is running. Use POST /api with 'questions_file' and optional 'data_file'.",
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
