
"""
Main pipeline for Term Sheet extraction
- Walk MAIN_FOLDER, find all PDFs
- Extract chunks -> parse -> write to Excel
- Each PDF corresponds to one row in EXPORT sheet
"""

import os
from extractor import extract_chunks_from_termsheet
from parser import parse_with_llm_gemini
from parser import parse_with_llm
from writer import write_to_excel  
from config import MAIN_FOLDER, GEMINI_MODEL, GROQ_MODEL, TOP_K, CHUNK_SIZE, OVERLAP, PROMPTS_FILE




def find_all_pdfs(folder_path: str):
    """Return full paths of all PDFs in folder (non-recursive)."""
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith(".pdf")]


def main():
    pdf_paths = find_all_pdfs(MAIN_FOLDER)
    if not pdf_paths:
        print(f"No PDFs found in {MAIN_FOLDER}")
        return

    for pdf_path in sorted(pdf_paths):
        pdf_name = os.path.basename(pdf_path)
        print(f"Processing Term Sheet: {pdf_name}")

        # Extract text and create chunks
        chunks = extract_chunks_from_termsheet(pdf_path,
                                                chunk_size=CHUNK_SIZE,
                                                overlap=OVERLAP,
                                                folder_name=os.path.splitext(pdf_name)[0])

        # Parse chunks with Gemini
        results = parse_with_llm_gemini(chunks, PROMPTS_FILE, gemini_model=GEMINI_MODEL, top_k=TOP_K)
        
        #Parse chunks with Groq
        #results = parse_with_llm(chunks, PROMPTS_FILE,groq_model=GROQ_MODEL,top_k=TOP_K)

        # Write each parsed result into Excel (one row per PDF)
        for r in results:
            run_for = r.get("run_for")
            json_result = r.get("result", {})
            if run_for and isinstance(json_result, dict):
                write_to_excel(json_result)

    print("✅ Term Sheet processing completed.")


if __name__ == "__main__":
    main()
