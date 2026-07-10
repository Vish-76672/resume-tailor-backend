import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import pdfplumber
import docx

app = Flask(__name__)
CORS(app)  # allow frontend (Vercel) to call this backend cross-origin

# --- Config ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-3-flash-preview")


# --- Resume text extraction ---
def extract_text_from_pdf(file_stream):
    text = ""
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_text_from_docx(file_stream):
    document = docx.Document(file_stream)
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def extract_text(file_storage):
    filename = file_storage.filename.lower()
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file_storage.stream)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file_storage.stream)
    elif filename.endswith(".txt"):
        return file_storage.stream.read().decode("utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file type. Upload PDF, DOCX, or TXT.")


# --- Gemini prompt ---
def build_prompt(resume_text, job_description):
    return f"""You are a resume optimization expert and ATS (Applicant Tracking System) simulator.
You will be given a candidate's resume and a job description.

Do NOT fabricate any experience, skills, or facts not present in the original resume.
Only reorder, reword, and re-prioritize existing content to match the JD's language.

Return ONLY valid JSON (no markdown fences, no preamble, no trailing commas) with exactly these keys:

{{
  "match_score": <integer 0-100, your estimate of how well the ORIGINAL resume matches the JD before tailoring>,
  "matched_keywords": ["keyword1", "keyword2", ...up to 12 keywords/skills from the JD that ARE present in the resume],
  "missing_keywords": ["keyword1", "keyword2", ...up to 10 important keywords/skills from the JD that are NOT present in the resume],
  "tailored_resume": "the full tailored resume as plain text, formatted with clear section headers (SUMMARY, EXPERIENCE, SKILLS, EDUCATION, etc.)",
  "cover_letter": "a 150-word cover letter tailored to this specific role",
  "changes_summary": ["bullet point 1 describing a specific change made and why", "bullet point 2", "..."]
}}

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}
"""


def call_gemini(prompt):
    response = model.generate_content(prompt)
    raw_text = response.text.strip()
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)
    return json.loads(raw_text)


# --- Routes ---
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/list-models", methods=["GET"])
def list_models():
    """Diagnostic route: shows exactly which models this API key/project can call.
    Safe to remove later, but useful whenever Gemini's model lineup changes and
    causes 404s on whatever model name is hardcoded above."""
    try:
        models = genai.list_models()
        available = [
            m.name
            for m in models
            if "generateContent" in m.supported_generation_methods
        ]
        return jsonify({"available_models": available})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tailor", methods=["POST"])
def tailor():
    try:
        if "resume" not in request.files:
            return jsonify({"error": "No resume file uploaded"}), 400

        resume_file = request.files["resume"]
        job_description = request.form.get("jd", "").strip()

        if not job_description:
            return jsonify({"error": "Job description is required"}), 400

        resume_text = extract_text(resume_file)
        if not resume_text:
            return jsonify({"error": "Could not extract text from resume. Try a different file."}), 400

        prompt = build_prompt(resume_text, job_description)
        result = call_gemini(prompt)

        # defensive defaults in case Gemini omits a key
        result.setdefault("match_score", 0)
        result.setdefault("matched_keywords", [])
        result.setdefault("missing_keywords", [])
        result.setdefault("changes_summary", [])

        return jsonify(result)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except json.JSONDecodeError:
        return jsonify({"error": "AI response could not be parsed. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)