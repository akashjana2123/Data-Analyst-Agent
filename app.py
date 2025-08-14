
import os
import networkx as nx
import re
import json
import base64
import tempfile
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Optional image conversion
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# LangChain / LLM imports (keep as you used)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TDS Data Analyst Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML interface"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1><p>Please ensure index.html is in the same directory as app.py</p>", status_code=404)


LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 180))

# -----------------------------
# Tools
# -----------------------------

@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """
    Fetch a URL and return data as a DataFrame (supports HTML tables, CSV, Excel, Parquet, JSON, and plain text).
    Always returns {"status": "success", "data": [...], "columns": [...]} if fetch works.
    """
    print(f"Scraping URL: {url}")
    try:
        from io import BytesIO, StringIO
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.google.com/",
        }

        resp = requests.get(url, headers=headers, timeout=50)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()

        df = None

        # --- CSV ---
        if "text/csv" in ctype or url.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(resp.content))

        # --- Excel ---
        elif any(url.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in ctype:
            df = pd.read_excel(BytesIO(resp.content))

        # --- Parquet ---
        elif url.lower().endswith(".parquet"):
            df = pd.read_parquet(BytesIO(resp.content))

        # --- JSON ---
        elif "application/json" in ctype or url.lower().endswith(".json"):
            try:
                data = resp.json()
                df = pd.json_normalize(data)
            except Exception:
                df = pd.DataFrame([{"text": resp.text}])

        # --- HTML / Fallback ---
        elif "text/html" in ctype or re.search(r'/wiki/|\.org|\.com', url, re.IGNORECASE):
            html_content = resp.text
            # Try HTML tables first
            try:
                tables = pd.read_html(StringIO(html_content), flavor="bs4")
                if tables:
                    df = tables[0]
            except ValueError:
                pass

            # If no table found, fallback to plain text
            if df is None:
                soup = BeautifulSoup(html_content, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                df = pd.DataFrame({"text": [text]})

        # --- Unknown type fallback ---
        else:
            df = pd.DataFrame({"text": [resp.text]})

        # --- Normalize columns ---
        df.columns = df.columns.map(str).str.replace(r'\[.*\]', '', regex=True).str.strip()

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist()
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# -----------------------------
# Utilities for executing code safely
# -----------------------------
def clean_llm_output(output: str) -> Dict:
    """
    Extract JSON object from LLM output robustly.
    Returns dict or {"error": "..."}
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
    # if already under limit, return png data uri
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
    # as last resort return a downsized PNG even if > max_bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    buf.seek(0)
    return  base64.b64encode(buf.getvalue()).decode('ascii')
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


# -----------------------------
# LLM agent setup
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_MODEL", "gemini-2.5-pro"),
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Tools list for agent (LangChain tool decorator returns metadata for the LLM)
tools = [scrape_url_to_dataframe]  # we only expose scraping as a tool; agent will still produce code

# Prompt: instruct agent to call the tool and output JSON only
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a full-stack autonomous data analyst agent.

You will receive:
- A set of **rules** for this request (these rules may differ depending on whether a dataset is uploaded or not)
- One or more **questions**
- An optional **dataset preview**

You must:
1. Follow the provided rules exactly.
2. Return only a valid JSON object — no extra commentary or formatting.
3. The JSON must contain:
   - "questions":  keys provided in the questions file
   - "code": "..." (Python code that fills `results` with exact type of answer of each question as given in questions file and question keys as keys)\n'
   - "Note" : the type of each answer should match the type it is asked in question file(e.g. int, float, str, boolean ,base64).
4. Your Python code will run in a sandbox with:
   - pandas, numpy, matplotlib available
   - A helper function `plot_to_base64(max_bytes=100000)` for generating base64-encoded images under 100KB.
5. When returning plots, always use `plot_to_base64()` to keep image sizes small.
6. Make sure all variables are defined before use, and the code can run without any undefined references.
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(
    llm=llm,
    tools=[scrape_url_to_dataframe],  # let the agent call tools if it wants; we will also pre-process scrapes
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


from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
import os, json, tempfile
from io import BytesIO
import pandas as pd
import numpy as np

from fastapi import Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO, StringIO
import tempfile, json
import pandas as pd

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
import os, json, tempfile
from io import BytesIO
import pandas as pd
import numpy as np

@app.post("/api")
async def analyze_data(request: Request):
    try:
        form = await request.form()

        # Collect all uploaded files from the form
        uploads = []
        for _, value in form.multi_items():
            if isinstance(value, UploadFile):
                uploads.append(value)

        if not uploads:
            raise HTTPException(400, "Upload at least one file (.txt questions file is required).")

        # Find exactly one .txt as questions file
        txt_files = [f for f in uploads if (f.filename or "").lower().endswith(".txt")]
        if len(txt_files) != 1:
            raise HTTPException(400, "Exactly one .txt questions file is required.")
        questions_file = txt_files[0]
        raw_questions = (await questions_file.read()).decode("utf-8")


        type_map = {}
        patterns = [
            re.compile(r"^\s*-\s*`([^`]+)`\s*:\s*([a-zA-Z ]+)", re.MULTILINE),  # bullet list with backticks
            re.compile(r"^\s*-\s*([a-zA-Z0-9_]+)\s*:\s*([a-zA-Z ]+)", re.MULTILINE),  # bullet list no backticks
            re.compile(r"`([^`]+)`\s*\((number|string|boolean|base64)[s]?\)", re.IGNORECASE),  # inline (type)
        ]

        for pat in patterns:
            for match in pat.finditer(raw_questions):
                key, type_hint = match.groups()
                norm_type = type_hint.strip().lower()
                if norm_type in ("number", "float", "int"):
                    norm_type = "number"
                elif norm_type in ("string", "str"):
                    norm_type = "string"
                elif "base64" in norm_type:
                    norm_type = "base64"
                elif norm_type in ("bool", "boolean"):
                    norm_type = "boolean"
                type_map[key.strip()] = norm_type

        # Build type note for LLM prompt
        type_note = "\nNote: The following are the exact expected types for each key based on the questions file:\n"
        if type_map:
            for k, t in type_map.items():
                type_note += f"- {k}: {t}\n"
        else:
            type_note += "(No explicit types detected in questions file)\n"

        # All others are candidate datasets
        data_candidates = [f for f in uploads if f is not questions_file]

        pickle_path = None
        df_preview = ""
        dataset_uploaded = False

        # Try to parse the first valid dataset
        for data_file in data_candidates:
            filename = (data_file.filename or "").lower()
            content = await data_file.read()

            try:
                if filename.endswith(".csv"):
                    df = pd.read_csv(BytesIO(content))
                elif filename.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(BytesIO(content))
                elif filename.endswith(".parquet"):
                    df = pd.read_parquet(BytesIO(content))
                elif filename.endswith(".json"):
                    try:
                        df = pd.read_json(BytesIO(content))
                    except ValueError:
                        df = pd.DataFrame(json.loads(content.decode("utf-8")))
                elif filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                    try:
                        if PIL_AVAILABLE:
                            image = Image.open(BytesIO(content))
                            image = image.convert("RGB")  # ensure RGB format
                            df = pd.DataFrame({"image": [image]})  # store as a single column DataFrame
                        else:
                            raise ValueError("PIL is not available for image processing.")
                    except Exception as e:
                        raise ValueError(f"Failed to process image file: {str(e)}")
                else:
                    continue  # unsupported type
            except Exception:
                continue  # failed parse

            # If parsed successfully
            dataset_uploaded = True
            temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            temp_pkl.close()
            df.to_pickle(temp_pkl.name)
            pickle_path = temp_pkl.name

            df_preview = (
                f"\n\nThe uploaded dataset has {len(df)} rows and {len(df.columns)} columns.\n"
                f"Columns: {', '.join(df.columns.astype(str))}\n"
                f"First rows:\n{df.head(5).to_markdown(index=False)}\n"
            )
            break  # only the first valid dataset is used

        # Build LLM rules
        if dataset_uploaded:
            llm_rules = (
                "Rules:\n"
                "1) You have access to a pandas DataFrame called `df` and its dictionary form `data`.\n"
                "2) DO NOT call scrape_url_to_dataframe() or fetch any external data.\n"
                "3) Use only the uploaded dataset for answering questions.\n"
                "4) Produce a final JSON object with keys:\n"
                '   - "questions": exact keys provided in questions txt file \n'
                '   - "code": "..."  (Python code that fills `results` with exact type of answer of each question as given in questions file and question keys as keys)\n'
                "5) For plots: use plot_to_base64() helper to return base64 image data under 100kB.\n"
            )
        else:
            llm_rules = (
                "Rules:\n"
                "1) If you need web data, CALL scrape_url_to_dataframe(url).\n"
                "2) Produce a final JSON object with keys:\n"
                '   - "questions": exact keys provided in questions txt file \n'
                '   - "code": "..."  (Python code that fills `results` with exact type of answer of each question as given in questions file and question keys as keys)\n'
                "3) For plots: use plot_to_base64() helper to return base64 image data under 100kB.\n"
            )

        llm_input = (
            f"{llm_rules}\nQuestions:\n{raw_questions}\n"
            f"{df_preview if df_preview else ''}\n"
            f"{type_note}\n"
            "Respond with the JSON object only."
        )

        # Run unified agent
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            fut = ex.submit(run_agent_safely_unified, llm_input, pickle_path, type_map)
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


# -----------------------------
# Runner: orchestrates agent -> pre-scrape inject -> execute
# -----------------------------
    
def run_agent_safely_unified(llm_input: str, pickle_path: str = None, type_map: dict = None) -> Dict:
    """
    Runs the LLM agent and executes code.
    - If pickle_path is provided, injects that DataFrame directly.
    - If no pickle_path, falls back to scraping when needed.
    """
    try:
        response = agent_executor.invoke({"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS})
        raw_out = response.get("output") or response.get("final_output") or response.get("text") or ""
        if not raw_out:
            return {"error": "Agent returned no output"}

        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            return parsed

        if "code" not in parsed or "questions" not in parsed:
            return {"error": f"Invalid agent response: {parsed}"}

        code = parsed["code"]
        questions = parsed["questions"]

        # If no pickle provided, check if code tries to scrape
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

        # Execute code with pickle injection if available
        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
        if exec_result.get("status") != "success":
            return {"error": f"Execution failed: {exec_result.get('message')}", "raw": exec_result.get("raw")}

        results_dict = exec_result.get("result", {})
        print(f"Results dict: {results_dict}")
        if type_map:
            for k, expected in type_map.items():
                if k not in results_dict:
                    continue
                v = results_dict[k]
                if expected == "number":
                    try:
                        num = float(v)
                        results_dict[k] = int(num) if num.is_integer() else num
                    except:
                        results_dict[k] = None
                elif expected in ("string", "base64"):
                    results_dict[k] = "" if v is None else str(v)
                elif expected == "boolean":
                    results_dict[k] = bool(v)  
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

