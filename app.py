import streamlit as st
import os
import json
import pandas as pd
import tempfile
from io import BytesIO
import json
import pandas as pd
# Import your existing modules
# We wrap these in try-except to handle potential import errors gracefully in the UI
try:
    from extractor import extract_chunks_from_termsheet
    from parser import parse_with_llm_gemini, parse_with_llm
    from config import CHUNK_SIZE, OVERLAP, PROMPTS_FILE, TOP_K, GEMINI_MODEL, GROQ_MODEL
except ImportError as e:
    st.error(f"Error importing modules: {e}. Make sure extractor.py, parser.py, and config.py are in the same directory.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Term Sheet Extractor", layout="wide")

st.title("📄 AI Term Sheet Parser")
st.markdown("Upload PDF Term Sheets to extract structured data using Gemini or Groq.")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    # Model Selection
    model_provider = st.radio("Select Provider", ["Gemini", "Groq"])
    
    # API Key Handling
    if model_provider == "Gemini":
        api_key = st.text_input("Gemini API Key", type="password")
        model_name = st.text_input("Model Name", value=GEMINI_MODEL)
        os.environ["GEMINI_API_KEY"] = api_key # Set environment variable for parser.py
    else:
        api_key = st.text_input("Groq API Key", type="password")
        model_name = st.text_input("Model Name", value=GROQ_MODEL)
        os.environ["GROQ_API_KEY"] = api_key

    st.divider()
    
    # Hyperparameters
    st.subheader("Advanced Settings")
    chunk_size = st.number_input("Chunk Size", value=CHUNK_SIZE)
    overlap = st.number_input("Overlap", value=OVERLAP)
    top_k = st.number_input("Top K Context", value=TOP_K)
    
    st.info(f"Using Prompts from: `{PROMPTS_FILE}`")

# --- Main Interface ---

uploaded_files = st.file_uploader("Upload Term Sheet PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("Start Extraction", type="primary", disabled=not uploaded_files):
    
    if not api_key:
        st.warning("Please enter an API Key in the sidebar to proceed.")
        st.stop()

    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing: {uploaded_file.name}...")
        
        # 1. Save uploaded file to a temporary file 
        # (Necessary because extractor.py expects a file path for pdfplumber)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # 2. Extract Chunks
            chunks = extract_chunks_from_termsheet(
                tmp_file_path,
                chunk_size=chunk_size,
                overlap=overlap,
                folder_name=uploaded_file.name
            )

            # 3. Parse with LLM
            if model_provider == "Gemini":
                # Ensure parser.py uses the key from os.environ
                parsed_data = parse_with_llm_gemini(chunks, PROMPTS_FILE, gemini_model=model_name, top_k=top_k)
            else:
                parsed_data = parse_with_llm(chunks, PROMPTS_FILE, groq_model=model_name, top_k=top_k)

            # 4. Flatten results for the DataFrame
            for item in parsed_data:
                # The result is inside item['result']
                row_data = item.get("result", {})
                
                # If the result is just a wrapper, try to extract the inner dict
                if isinstance(row_data, str):
                    # Fallback if parsing failed deeply
                    row_data = {"raw_output": row_data}
                
                # Add metadata
                row_data["Source File"] = uploaded_file.name
                all_results.append(row_data)

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

        # Update Progress
        progress_bar.progress((idx + 1) / len(uploaded_files))

    status_text.text("Processing Complete!")
    
    # --- Results Display ---
    if all_results:
        st.subheader("Extracted Data")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Reorder columns to put "Source File" first if possible
        cols = ["Source File"] + [c for c in df.columns if c != "Source File"]
        df = df[cols]
        # --- Normalize DataFrame to avoid pyarrow ArrowInvalid (mixed list vs scalar) ---


        def normalize_cell_for_arrow(x):
            # convert list/tuple -> comma-joined string (if possible) otherwise JSON
            if isinstance(x, (list, tuple)):
                # if elements are simple scalars, join them; else use json.dumps
                simple = all(not isinstance(i, (list, tuple, dict)) for i in x)
                if simple:
                    try:
                        return ", ".join("" if i is None else str(i) for i in x)
                    except Exception:
                        return json.dumps(x, ensure_ascii=False)
                else:
                    return json.dumps(x, ensure_ascii=False)
            # convert dict -> json string
            if isinstance(x, dict):
                return json.dumps(x, ensure_ascii=False)
            # convert NaN/None -> empty string
            if pd.isna(x):
                return ""
            # leave simple scalars as-is, otherwise stringify
            if isinstance(x, (str, int, float, bool)):
                return x
            return str(x)

        # Detect columns that contain list/tuple/dict values
        problem_cols = [c for c in df.columns if df[c].apply(lambda v: isinstance(v, (list, tuple, dict))).any()]

        if problem_cols:
            # optional debug: show names and some sample values for inspection
            st.warning(f"Normalizing columns for Arrow compatibility: {problem_cols}")
            for c in problem_cols:
                try:
                    st.write(f"Samples from `{c}`:", df[c].dropna().head(5).tolist())
                except Exception:
                    pass

            # Apply normalization to only the problem columns for performance
            for c in problem_cols:
                df[c] = df[c].apply(normalize_cell_for_arrow)

        # Additional safety: convert any remaining object-typed column to strings if PyArrow might still complain
        for c in df.select_dtypes(include=['object']).columns:
            # if column still contains non-string types, coerce to string but preserve empty string for NaN
            if not df[c].map(lambda v: isinstance(v, str)).all():
                df[c] = df[c].apply(lambda v: "" if pd.isna(v) else str(v))

        st.dataframe(df)

        # --- Excel Download ---
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='EXPORT')
            
        st.download_button(
            label="📥 Download Excel Report",
            data=buffer.getvalue(),
            file_name="TermSheet_Output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("No data extracted. Please check the PDF content or Prompts.")