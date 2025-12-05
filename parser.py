"""
parser.py
- Loads prompts from Prompts/prompts_spo_framework.json
- Builds a TF-IDF index over extracted chunks
- For each prompt, filter chunks by "run_for" (framework/spo/both),
  retrieve top_k chunks, create system/user message,
  and call Groq LLM to produce the output JSON.
- Exports a list of parsed JSONs (one per prompt).
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import time
import re

from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()



# ---------------- TF-IDF Retrieval ---------------- #

def build_tfidf_index(chunks: List[Dict]) -> Dict:
    """Return vectorizer and matrix for search, plus the chunk texts."""
    texts = [c["chunk"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    if texts:
        matrix = vectorizer.fit_transform(texts)
    else:
        matrix = None
    return {"vectorizer": vectorizer, "matrix": matrix, "texts": texts}


def retrieve_top_k(query: str, index: Dict, k: int = 5) -> List[int]:
    """Return top-k indices (into index['texts']) most similar to query."""
    if index["matrix"] is None:
        return []
    qv = index["vectorizer"].transform([query])
    sims = cosine_similarity(qv, index["matrix"]).flatten()
    topk_idx = np.argsort(-sims)[:k]
    return [int(i) for i in topk_idx if sims[i] > 0]


def assemble_context(chunks: List[Dict], top_indices: List[int]) -> str:
    """Create a human-readable context block with source and page metadata."""
    parts = []
    for i in top_indices:
        c = chunks[i]
        header = f"[source: {c.get('source','?')}] [page: {c.get('page','?')}] [chunk_idx: {c.get('chunk_index','?')}]"
        parts.append(header + "\n" + c["chunk"])
    return "\n\n---\n\n".join(parts)


# ---------------- Groq API ---------------- #

def call_groq(model: str, messages: List[Dict], temperature: float = 0.0, max_retries: int = 3) -> Dict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in environment.")

    # some versions use Groq(api_key=...), others just Groq()
    client = Groq(api_key=api_key) if hasattr(Groq, "__call__") else Groq()

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(1.0 * attempt)


# ---------------- Core Parsing Logic ---------------- #

def parse_with_llm(chunks: List[Dict],prompts_path: str,groq_model: str ,top_k: int = 5) -> List[Dict]:
    """
    chunks: list of dicts from extractor.py
    prompts_path: path to prompts_spo_frameworks.json
    groq_model: Groq model name
    returns: list of dicts { "prompt_id": ..., "result": <parsed json> }
    """

    # load prompts
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    results = []

    for p in prompts:
        run_for = p.get("run_for", "both").lower()

        # Filter chunks based on run_for
        if run_for == "termsheet":
            relevant_chunks = [c for c in chunks if c.get("source") == "termsheet"]
        else:  # "both" or missing
            relevant_chunks = chunks

        # build TF-IDF index for the filtered set
        index = build_tfidf_index(relevant_chunks)

        query = p.get("instruction") or p.get("query") or ""
        top_idx = retrieve_top_k(query, index, k=top_k)
        context = assemble_context(relevant_chunks, top_idx)

        system_msg = {
            "role": "system",
            "content": (
                "You are a JSON extraction assistant. Use ONLY the provided CONTEXT to answer. "
                "Output must be valid JSON and must match the provided schema or example. "
                "If a field cannot be found in the context, set it to null or an empty string."
            )
        }

        user_content = (
            "CONTEXT:\n\n"
            f"{context}\n\n"
            "INSTRUCTION:\n\n"
            f"{p['instruction']}\n\n"
            "OUTPUT_SCHEMA / EXAMPLE:\n\n"
            f"{json.dumps(p['json_schema'], indent=2)}\n\n"
            "Return ONLY the JSON (no extra commentary)."
        )
        user_msg = {"role": "user", "content": user_content}

        resp = call_groq(model=groq_model, messages=[system_msg, user_msg], temperature=0.0)

        try:
            content = resp.choices[0].message.content
        except Exception:
            content = str(resp)

        # try parsing JSON
        try:
            parsed = json.loads(content)
        except Exception:
            m = re.search(r'(\{.*\}|\[.*\])', content, flags=re.S)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except Exception:
                    parsed = {"_raw": content}
            else:
                parsed = {"_raw": content}

        results.append({
            "prompt_id": p.get("id"),
            "run_for": run_for,
            "result": parsed,
            "used_context_indices": top_idx,
            "raw_model_output": content
        })

    return results

#Gemini Parsing

def parse_with_llm_gemini(chunks: List[Dict],prompts_path: str,gemini_model: str,top_k: int = 5) -> List[Dict]:
    """
    chunks: list of dicts from extractor.py
    prompts_path: path to prompts.json
    gemini_model: Gemini model name (e.g., "gemini-1.5-flash")
    returns: list of dicts { "prompt_id": ..., "result": <parsed json> }
    """

    # load prompts
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    results = []

    for p in prompts:
        run_for = p.get("run_for", "both").lower()

        # Filter chunks based on run_for
        if run_for == "termsheet":
            relevant_chunks = [c for c in chunks if c.get("source") == "termsheet"]
        else: 
            relevant_chunks = chunks

        # build TF-IDF index for the filtered set
        index = build_tfidf_index(relevant_chunks)

        query = p.get("instruction") or p.get("query") or ""
        top_idx = retrieve_top_k(query, index, k=top_k)
        context = assemble_context(relevant_chunks, top_idx)

        system_msg = {
            "role": "system",
            "content": (
                "You are a JSON extraction assistant. Use ONLY the provided CONTEXT to answer. "
                "Output must be valid JSON and must match the provided schema or example. "
                "If a field cannot be found in the context, set it to null or an empty string."
            )
        }

        user_content = (
            "CONTEXT:\n\n"
            f"{context}\n\n"
            "INSTRUCTION:\n\n"
            f"{p['instruction']}\n\n"
            "OUTPUT_SCHEMA / EXAMPLE:\n\n"
            f"{json.dumps(p['json_schema'], indent=2)}\n\n"
            "Return ONLY the JSON (no extra commentary)."
        )
        user_msg = {"role": "user", "content": user_content}

        resp = call_gemini(model_gemini=gemini_model,
                           messages=[system_msg, user_msg],
                           temperature=0.0)
        
        # print(resp.usage_metadata)

        try:
            content = resp.text
        except Exception:
            content = str(resp)

        # try parsing JSON
        try:
            parsed = json.loads(content)
        except Exception:
            m = re.search(r'(\{.*\}|\[.*\])', content, flags=re.S)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except Exception:
                    parsed = {"_raw": content}
            else:
                parsed = {"_raw": content}

        results.append({
            "prompt_id": p.get("id"),
            "run_for": "termsheet",
            "result": parsed,
            "used_context_indices": top_idx,
            "raw_model_output": content
        })

    return results

#Call Gemini

def call_gemini(model_gemini: str,
                messages: List[Dict],
                temperature: float = 0.0,
                max_retries: int = 3) -> Dict:
    """
    Call Gemini chat model with retries.
    messages: list of {"role": "system"|"user"|"assistant", "content": str}
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in environment.")

    client = genai.Client(api_key=api_key)

    for attempt in range(1, max_retries + 1):
        try:
            # Gemini doesn’t use role-based messages directly like OpenAI.
            # We'll stitch them together into a prompt string.
            prompt = ""
            for m in messages:
                role = m["role"].upper()
                prompt += f"{role}: {m['content']}\n\n"

            # model_client = genai.GenerativeModel(model)
            # response = model_client.generate_content(
            #     prompt,
            #     generation_config={"temperature": temperature}
            # )
            # return response

            user_messages = [m["content"] for m in messages if m["role"] == "user"]
            system_messages = [m["content"] for m in messages if m["role"] == "system"]


            response = client.models.generate_content(
                model = model_gemini,
                contents = user_messages,

                config = types.GenerateContentConfig(
                    system_instruction = system_messages,
                    temperature = temperature,

                )

            )

            return response
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(1.0 * attempt)

