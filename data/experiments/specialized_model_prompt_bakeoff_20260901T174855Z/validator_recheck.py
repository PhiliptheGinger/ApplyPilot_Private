"""Run the actual worst fabrications found in the bake-off through the
NOW-FIXED validate_json_fields, to check whether check_unsupported_skill_
claims and check_date_fabrication catch them for real -- not synthetic
reproductions, the real raw model output, hand-transcribed into the JSON
contract shape (title/summary/skills/experience/projects/education) since
this experiment's native/plan configs produced markdown resume TEXT, not
JSON (production's real pipeline always works with JSON). Content below is
copied verbatim from raw_output_qwen25_control.json / raw_output_iteratecv.json
/ raw_output_qwen3.json in this same directory.

No production code modified by this script. Read-only check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config  # noqa: E402
from applypilot.scoring.validator import validate_json_fields  # noqa: E402

profile = config.load_profile()

CASES = []

# --- Case 1: qwen2.5:3b / job 22916 / native -- fabricated an entire prior
# job "Hotel Partner Solutions Specialist at Expedia Group" (the exact
# target company/title) with duties lifted from the job posting.
CASES.append({
    "label": "qwen2.5:3b / 22916 / native -- fabricated 'Hotel Partner Solutions Specialist at Expedia Group'",
    "data": {
        "title": "Software Development Engineer",
        "summary": "Experienced in Python programming with a strong foundation in software development principles. Skilled in building scalable solutions and conducting data analysis to improve user experiences. Effective communicator with experience in customer-facing roles and public speaking.",
        "skills": {"Languages": "Python", "Tools": "Web Development (HTML, CSS, JavaScript), Data Analysis, Optical Character Recognition (OCR)"},
        "experience": [
            {
                "header": "Hotel Partner Solutions Specialist at Expedia Group",
                "subtitle": "",
                "bullets": [
                    "Assisted with issue or incident investigations by gathering and documenting necessary information.",
                    "Identified, researched, tested, and escalated bugs to the appropriate team for resolution.",
                    "Translated data analysis and findings into accessible visuals to identify chronic issues or product opportunities.",
                    "Applied technical expertise to resolve service and system disruptions in coordination with technical teams.",
                ],
            },
        ],
        "projects": [
            {"header": "Standup-OCR - Optical Character Recognition Tooling and Image Processing", "subtitle": "", "bullets": ["Developed Python scripts for optical character recognition and transcription of images and documents."]},
        ],
        "education": "University of North Carolina at Greensboro\nBachelor of Arts in Media Studies\nGraduated: 2022",
    },
})

# --- Case 2: qwen2.5:3b / job 22916 / plan -- renamed "Alignment
# Technician" to "Translator at Mavis" and invented a wholly separate
# "Data Analyst at UST Logistics" role split out of the real Installer entry.
CASES.append({
    "label": "qwen2.5:3b / 22916 / plan -- 'Translator at Mavis' + fabricated 'Data Analyst at UST Logistics'",
    "data": {
        "title": "Technical Support & Customer Service",
        "summary": "Experienced in Python programming with a strong foundation in software development principles.",
        "skills": {"Languages": "Python"},
        "experience": [
            {"header": "Sales Consultant at AMP Smart", "subtitle": "Sales | Dates not specified", "bullets": ["Developed and implemented targeted outreach and sales strategies for residential solar solutions, increasing qualified lead identification."]},
            {"header": "Translator at Mavis (National Tire and Battery)", "subtitle": "Technical Support | Dates not specified", "bullets": ["Performed vehicle alignment and maintenance tasks including tires, shocks, struts, brakes, and fluids."]},
            {"header": "Data Analyst at UST Logistics", "subtitle": "Data Analysis | Dates not specified", "bullets": ["Conducted data analysis and findings to identify customer and employee satisfaction trends.", "Created accessible visuals to communicate data insights and highlight chronic issues or product opportunities."]},
            {"header": "Installer at Alex Prosperity Group / UST Logistics", "subtitle": "Installation | Dates not specified", "bullets": ["Installed home appliances contracted through Lowe's with attention to detail and customer satisfaction."]},
        ],
        "projects": [],
        "education": "University of North Carolina at Greensboro\nBachelor of Arts in Media Studies\n2019-2022",
    },
})

# --- Case 3: IterateCV / job 7267 / native -- fabricated an entire
# fictional "ALIGNMENT TECHNIQUES" experience entry plus an explicit
# "over 5 years of experience in software engineering" claim.
CASES.append({
    "label": "IterateCV / 7267 / native -- fabricated 'ALIGNMENT TECHNIQUES' role + '5+ years' claim",
    "data": {
        "title": "Software Development Engineer",
        "summary": "Dedicated Software Development Engineer with over 5 years of experience in software engineering. Proven ability to plan, design, develop, and test software systems, with a strong foundation in Python programming and data analysis.",
        "skills": {"Languages": "Python", "Tools": "Data Analysis, Machine Learning, RESTful Services, Databases (SQL), Agile Software Development"},
        "experience": [
            {"header": "Stand-up Comedian", "subtitle": "Public Speaking | Dates not specified", "bullets": ["Delivered live performances requiring clear verbal communication and audience engagement."]},
            {"header": "ALIGNMENT TECHNIQUES", "subtitle": "", "bullets": ["Conducted data analysis and experimentation to improve software performance and user experience.", "Built RESTful services to support scalable and high-performance solutions.", "Utilized databases and SQL for data management and validation."]},
        ],
        "projects": [
            {"header": "Stand-up-OCR - OCR Tooling and Image Processing", "subtitle": "Python | Dates not specified", "bullets": ["Developed Python scripts for optical character recognition and transcription of images and documents."]},
        ],
        "education": "University of North Carolina at Greensboro\nBachelor of Arts in Media Studies\n2019-2022",
    },
})

# --- Case 4: IterateCV / 7267 / plan -- systematic keyword-injection
# fabrication: appended "data"-related phrases onto nearly every bullet,
# including the Stand-up Comedian entry, none of it grounded.
CASES.append({
    "label": "IterateCV / 7267 / plan -- systematic 'data' keyword injection incl. Stand-up Comedian",
    "data": {
        "title": "Technical Support & Customer Service",
        "summary": "Support and troubleshooting background.",
        "skills": {"Languages": "Python"},
        "experience": [
            {"header": "Alignment Technician at National Tire and Battery / Mavis", "subtitle": "Technical Support | Dates not specified", "bullets": ["Performed vehicle alignment and maintenance tasks, which involved ensuring data accuracy and consistency in vehicle records."]},
            {"header": "Stand-up Comedian", "subtitle": "Public Speaking | Dates not specified", "bullets": ["Delivered live performances requiring clear verbal communication and audience engagement, utilizing data to inform comedic timing and audience connection."]},
        ],
        "projects": [],
        "education": "University of North Carolina at Greensboro\nBachelor of Arts in Media Studies\n2019-2022",
    },
})

# --- Case 5: qwen3:1.7b / job 27323 / native -- the single worst case:
# fabricated specific employment DATES for every entry (original says
# "Dates not specified" throughout) plus a fabricated skills list.
CASES.append({
    "label": "qwen3:1.7b / 27323 / native -- fabricated employment DATES + fabricated skills (the worst single case)",
    "data": {
        "title": "Software Development Engineer",
        "summary": "Experienced software developer with a strong foundation in software development principles.",
        "skills": {
            "Languages": "Python, JavaScript, HTML/CSS",
            "Tools": "Software development tools, version control (Git), data analysis tools (Pandas, NumPy), machine learning frameworks (Scikit-learn, TensorFlow), cloud platforms (AWS, Azure), data visualization (Tableau, Power BI)",
        },
        "experience": [
            {"header": "Alignment Technician at National Tire and Battery / Mavis", "subtitle": "Technical Support | 2019-2021", "bullets": ["Performed vehicle alignment and maintenance tasks including tires, shocks, struts, brakes, and fluids."]},
            {"header": "Package Handler at UPS", "subtitle": "Operations | 2018-2019", "bullets": ["Handled package sorting and logistics in a high-volume operational setting."]},
            {"header": "Installer at Alex Prosperity Group / UST Logistics", "subtitle": "Installation | 2017-2018", "bullets": ["Installed home appliances contracted through Lowe's with attention to detail and customer satisfaction."]},
        ],
        "projects": [],
        "education": "University of North Carolina at Greensboro\nBachelor of Arts in Media Studies\n2019-2022",
    },
})

results = []
for case in CASES:
    v = validate_json_fields(case["data"], profile, standup_decision="INCLUDE")
    results.append({"label": case["label"], "passed": v["passed"], "errors": v["errors"], "warnings": v["warnings"]})
    print(f"\n{'=' * 90}\n{case['label']}\n{'=' * 90}")
    print(f"passed: {v['passed']}")
    for e in v["errors"]:
        print(f"  ERROR: {e}")
    for w in v["warnings"]:
        print(f"  WARN:  {w}")

out_path = Path(__file__).resolve().parent / "validator_recheck_results.json"
out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\n\nSaved to {out_path}")
