import json
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd
import pymupdf
import requests
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Job Skill Helper",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None


st.html(
    """
    <style>
    header[data-testid="stHeader"] {
        display: none;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .stApp {
        background: #f6f8fc;
        color: #1f2937;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    p,
    span,
    label {
        color: #374151;
    }

    h1,
    h2,
    h3,
    h4 {
        color: #111827 !important;
    }

    .hero-box {
        background: linear-gradient(
            135deg,
            #ffffff 0%,
            #edf4ff 100%
        );
        border: 1px solid #dbe5f2;
        border-radius: 20px;
        padding: 28px 34px;
        margin-bottom: 28px;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.04);
    }

    .hero-title {
        font-size: 38px;
        font-weight: 750;
        color: #111827;
        margin-bottom: 6px;
    }

    .hero-text {
        font-size: 17px;
        color: #5b6472;
        margin: 0;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        color: #111827;
        margin-top: 6px;
        margin-bottom: 4px;
    }

    .section-description {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 18px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 1px solid #e3e8ef;
        border-radius: 16px;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.035);
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e3e8ef;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.035);
    }

    div[data-testid="stMetricLabel"] {
        color: #6b7280;
    }

    div[data-testid="stMetricValue"] {
        color: #111827;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 44px;
        transition: 0.2s ease;
    }

    .stButton > button[kind="primary"] {
        background: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #2563eb !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
        color: #ffffff !important;
    }

    .stButton > button[kind="secondary"] {
        background: #fff7ed !important;
        color: #c2410c !important;
        border: 1px solid #fb923c !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: #ffedd5 !important;
        color: #9a3412 !important;
        border-color: #f97316 !important;
    }

    .stButton > button[kind="secondary"]:disabled {
        background: #f3f4f6 !important;
        color: #6b7280 !important;
        border: 1px dashed #9ca3af !important;
        opacity: 1 !important;
        cursor: not-allowed;
    }

    div[data-baseweb="input"] {
        background: white !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"] input {
        color: #111827 !important;
        background: white !important;
        font-size: 15px !important;
    }

    div[data-baseweb="input"] input::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
        font-weight: 500 !important;
    }

    textarea {
        color: #111827 !important;
        background: white !important;
        border-radius: 10px !important;
    }

    textarea::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stTextArea"] label {
        color: #1f2937 !important;
        font-weight: 600 !important;
    }

    section[data-testid="stFileUploaderDropzone"] {
        background: #f9fbff;
        border-radius: 12px;
    }

    button[data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
    }

    div[data-testid="stExpander"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        margin-bottom: 9px;
    }

    .table-wrapper {
        width: 100%;
        overflow-x: auto;
        overflow-y: auto;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        background: #ffffff;
    }

    .theme-table {
        width: 100%;
        border-collapse: collapse;
        background: #ffffff;
        font-size: 14px;
    }

    .theme-table thead th {
        background: #eaf2ff;
        color: #1e3a5f;
        font-weight: 700;
        text-align: left;
        padding: 13px 15px;
        border-bottom: 1px solid #dbe5f2;
        position: sticky;
        top: 0;
        z-index: 1;
    }

    .theme-table tbody td {
        color: #374151;
        padding: 12px 15px;
        border-bottom: 1px solid #edf0f4;
        background: #ffffff;
    }

    .theme-table tbody tr:nth-child(even) td {
        background: #f8fafc;
    }

    .theme-table tbody tr:hover td {
        background: #eef6ff;
    }

    .theme-table tbody tr:last-child td {
        border-bottom: none;
    }

    .analysis-banner {
        background: linear-gradient(
            135deg,
            #eff6ff 0%,
            #f0fdf4 100%
        );
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }

    .analysis-banner-title {
        color: #111827;
        font-size: 25px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .analysis-banner-text {
        color: #64748b;
        font-size: 15px;
    }

    .url-help-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 13px 15px;
        margin-bottom: 14px;
        color: #1e3a5f;
        font-size: 14px;
        line-height: 1.6;
    }

    .url-help-title {
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 3px;
    }

    .manual-help {
        background: #f8fafc;
        border-left: 4px solid #94a3b8;
        padding: 9px 12px;
        border-radius: 6px;
        color: #475569;
        font-size: 13px;
        margin-bottom: 10px;
    }

    hr {
        border-color: #e5e7eb !important;
    }
    </style>
    """
)


st.html(
    """
    <div class="hero-box">
        <div class="hero-title">
            💼 Job Skill Helper
        </div>

        <div class="hero-text">
            Compare your resume with target jobs,
            discover missing skills and find what
            you should learn next.
        </div>
    </div>
    """
)


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


SKILLS = {
    "Python": ["python"],
    "Java": ["java"],
    "JavaScript": ["javascript", "js"],
    "TypeScript": ["typescript"],
    "React": ["react", "react.js", "reactjs"],
    "Angular": ["angular"],
    "Node.js": ["node.js", "nodejs", "node"],
    "Express.js": ["express.js", "expressjs", "express"],
    "Spring Boot": ["spring boot"],
    "FastAPI": ["fastapi"],
    "Flask": ["flask"],
    "MongoDB": ["mongodb", "mongo db"],
    "MySQL": ["mysql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "Redis": ["redis"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "Git": ["git"],
    "GitHub": ["github"],
    "REST APIs": [
        "rest api",
        "restful api",
        "rest apis",
        "restful apis",
    ],
    "GraphQL": ["graphql"],
    "Kafka": ["kafka", "apache kafka"],
    "Microservices": ["microservices", "microservice"],
    "CI/CD": [
        "ci/cd",
        "continuous integration",
        "continuous deployment",
    ],
    "Machine Learning": ["machine learning"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
}


LEARNING_GUIDE = {
    "Python": [
        "Learn Python basics and functions",
        "Practice object-oriented programming",
        "Build a small Python project",
    ],
    "Java": [
        "Learn Java syntax and OOP",
        "Practice collections and exception handling",
        "Build a small Java application",
    ],
    "JavaScript": [
        "Learn JavaScript fundamentals",
        "Practice arrays, objects and functions",
        "Build a small web project",
    ],
    "TypeScript": [
        "Learn TypeScript types and interfaces",
        "Practice TypeScript with JavaScript projects",
        "Build a small TypeScript application",
    ],
    "React": [
        "Learn components and props",
        "Practice state and hooks",
        "Build a small React application",
    ],
    "Angular": [
        "Learn Angular components and modules",
        "Practice services and routing",
        "Build a small Angular project",
    ],
    "Node.js": [
        "Learn Node.js fundamentals",
        "Practice creating backend APIs",
        "Build a small backend project",
    ],
    "Express.js": [
        "Learn Express routing",
        "Practice middleware and APIs",
        "Build a REST API using Express",
    ],
    "Spring Boot": [
        "Learn Spring Boot basics",
        "Practice controllers and services",
        "Build a simple REST API",
    ],
    "FastAPI": [
        "Learn FastAPI basics",
        "Practice creating API endpoints",
        "Build a simple backend API",
    ],
    "Flask": [
        "Learn Flask basics",
        "Practice routes and request handling",
        "Build a simple Flask application",
    ],
    "MongoDB": [
        "Learn MongoDB documents and collections",
        "Practice CRUD operations",
        "Connect MongoDB with a backend project",
    ],
    "MySQL": [
        "Learn SQL basics",
        "Practice SELECT, JOIN and GROUP BY",
        "Connect MySQL with a small application",
    ],
    "PostgreSQL": [
        "Learn PostgreSQL and SQL basics",
        "Practice tables, joins and queries",
        "Connect PostgreSQL with a backend application",
    ],
    "Redis": [
        "Learn Redis basics",
        "Understand caching",
        "Use Redis in a small backend project",
    ],
    "Docker": [
        "Learn containers and Docker images",
        "Practice writing a Dockerfile",
        "Containerize one of your projects",
    ],
    "Kubernetes": [
        "Learn pods and deployments",
        "Understand Kubernetes services",
        "Deploy a small containerized application",
    ],
    "AWS": [
        "Learn basic AWS services",
        "Understand EC2, S3 and IAM",
        "Deploy a small application on AWS",
    ],
    "Azure": [
        "Learn basic Azure services",
        "Understand cloud deployment",
        "Deploy a small application on Azure",
    ],
    "Git": [
        "Learn basic Git commands",
        "Practice branches and merging",
        "Manage a project using Git",
    ],
    "GitHub": [
        "Learn GitHub repositories",
        "Practice pull requests",
        "Upload and maintain projects on GitHub",
    ],
    "REST APIs": [
        "Understand HTTP methods",
        "Learn request and response handling",
        "Build and test a REST API",
    ],
    "GraphQL": [
        "Learn GraphQL queries and mutations",
        "Practice creating schemas",
        "Build a small GraphQL API",
    ],
    "Kafka": [
        "Learn Kafka producers and consumers",
        "Understand topics and messages",
        "Create a small event-driven demo",
    ],
    "Microservices": [
        "Understand microservice architecture",
        "Learn communication between services",
        "Split a small application into two services",
    ],
    "CI/CD": [
        "Understand continuous integration",
        "Learn basic automated pipelines",
        "Create a simple GitHub Actions workflow",
    ],
    "Machine Learning": [
        "Learn supervised and unsupervised learning",
        "Practice machine learning using Scikit-learn",
        "Build a small machine learning project",
    ],
    "TensorFlow": [
        "Learn TensorFlow basics",
        "Practice tensors and simple models",
        "Build a basic neural network",
    ],
    "PyTorch": [
        "Learn PyTorch tensors",
        "Practice creating simple models",
        "Build a basic neural network",
    ],
}


def show_table(dataframe, height=350):
    table_html = dataframe.to_html(
        index=False,
        classes="theme-table",
        border=0,
        escape=False,
    )

    st.html(
        f"""
        <div
            class="table-wrapper"
            style="max-height:{height}px;"
        >
            {table_html}
        </div>
        """
    )


def detect_skills(text):
    detected_skills = []
    text = text.lower()

    for skill, names in SKILLS.items():
        for name in names:
            pattern = r"(?<!\w)" + re.escape(name.lower()) + r"(?!\w)"

            if re.search(pattern, text):
                detected_skills.append(skill)
                break

    return detected_skills


def calculate_similarity(model, resume, job_description):
    embeddings = model.encode(
        [
            resume,
            job_description,
        ]
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]],
    )[0][0]

    similarity = max(0, min(float(similarity), 1))

    return round(similarity * 100, 1)


def get_priority(count, total_jobs):
    if count == total_jobs:
        return "High"

    if total_jobs == 3 and count == 2:
        return "Medium"

    if total_jobs == 2 and count == 1:
        return "Medium"

    return "Low"


def find_jobposting(data):
    if isinstance(data, dict):
        job_type = data.get("@type")

        if job_type == "JobPosting":
            return data

        if isinstance(job_type, list):
            if "JobPosting" in job_type:
                return data

        for value in data.values():
            result = find_jobposting(value)

            if result:
                return result

    elif isinstance(data, list):
        for item in data:
            result = find_jobposting(item)

            if result:
                return result

    return None


def fetch_job_description(url):
    try:
        url = url.strip()
        parsed_url = urlparse(url)

        if parsed_url.scheme not in ["http", "https"]:
            return False, "Please enter a valid http or https URL."

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=15,
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        json_scripts = soup.find_all(
            "script",
            type="application/ld+json",
        )

        for script in json_scripts:
            try:
                script_text = script.string

                if not script_text:
                    continue

                data = json.loads(script_text)
                job_data = find_jobposting(data)

                if job_data:
                    description = job_data.get("description")

                    if description:
                        description_soup = BeautifulSoup(
                            description,
                            "html.parser",
                        )

                        clean_description = description_soup.get_text(
                            separator="\n",
                            strip=True,
                        )

                        if len(clean_description) > 100:
                            return True, clean_description

            except Exception:
                continue

        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "noscript",
                "svg",
            ]
        ):
            tag.decompose()

        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.body
        )

        if main_content is None:
            return False, "Could not find readable content on this page."

        text = main_content.get_text(
            separator="\n",
            strip=True,
        )

        lines = []

        for line in text.splitlines():
            line = line.strip()

            if len(line) >= 3:
                lines.append(line)

        clean_text = "\n".join(lines)
        clean_text = clean_text[:30000]

        if len(clean_text) < 100:
            return False, "Could not find enough job description text."

        return True, clean_text

    except requests.exceptions.Timeout:
        return False, "The website took too long to respond."

    except requests.exceptions.RequestException:
        return (
            False,
            "The website could not be accessed. "
            "Please paste the job description manually.",
        )

    except Exception:
        return (
            False,
            "Unable to fetch the job description. "
            "Please paste it manually.",
        )


def job_input(job_number):
    text_key = f"job{job_number}_text"
    url_key = f"job{job_number}_url"
    button_key = f"job{job_number}_fetch"

    if text_key not in st.session_state:
        st.session_state[text_key] = ""

    input_col, description_col = st.columns(
        [0.38, 0.62],
        gap="large",
    )

    with input_col:
        st.markdown("### 🔗 Option 1 — Use Job URL")

        st.html(
            """
            <div class="url-help-box">
                <div class="url-help-title">
                    How to use the job link
                </div>

                <b>Step 1:</b> Paste the job posting URL below.<br>
                <b>Step 2:</b> Click
                <b>Fetch Job Description</b>.<br>
                <b>Step 3:</b> Check the fetched description
                shown on the right.
            </div>
            """
        )

        job_url = st.text_input(
            f"Paste Job {job_number} URL here",
            placeholder=(
                "Paste the complete job link here, "
                "for example: https://jobs.company.com/..."
            ),
            key=url_key,
            help=(
                "Copy the URL of the job posting "
                "from your browser and paste it here."
            ),
        )

        url_ready = bool(job_url.strip())

        if url_ready:
            fetch_button_text = (
                f"🔍 Fetch Job {job_number} Description"
            )
        else:
            fetch_button_text = (
                "⬆ Paste a URL above to enable Fetch"
            )

        fetch_clicked = st.button(
            fetch_button_text,
            key=button_key,
            width="stretch",
            type="secondary",
            disabled=not url_ready,
        )

        if fetch_clicked:
            with st.spinner("Reading the job page..."):
                success, result = fetch_job_description(job_url)

            if success:
                st.session_state[text_key] = result

                st.success(
                    "✅ Job description fetched! "
                    "Check it on the right."
                )
            else:
                st.warning(result)

        st.caption(
            "The Fetch button becomes active "
            "after you paste a URL."
        )

    with description_col:
        st.markdown("### 📝 Option 2 — Job Description")

        st.html(
            """
            <div class="manual-help">
                <b>You have two choices:</b><br>
                • If you used a URL, the fetched job description
                will appear here automatically.<br>
                • Or you can directly paste the job description
                here yourself.
            </div>
            """
        )

        job_description = st.text_area(
            f"Job Description {job_number}",
            height=250,
            placeholder=(
                "Fetched job description will appear here, "
                "or paste the job description manually..."
            ),
            key=text_key,
        )

    return job_description


st.html(
    """
    <div class="section-title">
        1. Add Your Resume
    </div>

    <div class="section-description">
        Upload your resume as a PDF or paste
        the resume text manually.
    </div>
    """
)

resume_left, resume_right = st.columns(
    [0.42, 0.58],
    gap="large",
)

resume_content = ""

with resume_left:
    with st.container(border=True):
        st.markdown("#### 📄 Resume Input")

        resume_file = st.file_uploader(
            "Upload PDF Resume",
            type=["pdf"],
        )

        st.markdown("**OR**")

        resume_text = st.text_area(
            "Paste Resume Text",
            height=160,
            placeholder="Paste your resume text here...",
        )


if resume_file is not None:
    try:
        resume_bytes = resume_file.getvalue()

        pdf = pymupdf.open(
            stream=resume_bytes,
            filetype="pdf",
        )

        for page in pdf:
            resume_content += page.get_text()

        pdf.close()

        with resume_right:
            with st.container(border=True):
                st.markdown("#### ✅ Resume Ready")

                st.success(resume_file.name)

                st.write(
                    "Your resume has been uploaded "
                    "and is ready for analysis."
                )

                with st.expander("👁 Preview Resume"):
                    preview_pdf = pymupdf.open(
                        stream=resume_bytes,
                        filetype="pdf",
                    )

                    for page_number, page in enumerate(preview_pdf):
                        pix = page.get_pixmap(
                            matrix=pymupdf.Matrix(1.1, 1.1)
                        )

                        image = pix.tobytes("png")

                        st.caption(
                            f"Page {page_number + 1}"
                        )

                        st.image(
                            image,
                            width="stretch",
                        )

                    preview_pdf.close()

    except Exception as error:
        with resume_right:
            st.error("Unable to read the PDF.")
            st.write(error)

elif resume_text.strip():
    resume_content = resume_text

    with resume_right:
        with st.container(border=True):
            st.markdown("#### ✅ Resume Ready")

            st.success(
                "Resume text added successfully."
            )

            st.write(
                "Your pasted resume text "
                "is ready for analysis."
            )

else:
    with resume_right:
        with st.container(border=True):
            st.markdown("#### Resume Status")

            st.info(
                "Upload a PDF or paste your "
                "resume text to continue."
            )


st.divider()


st.html(
    """
    <div class="section-title">
        2. Add Target Jobs
    </div>

    <div class="section-description">
        Add at least one target job.
        You can compare up to three jobs.
    </div>
    """
)

st.info(
    "💡 For each job: paste a job URL and click "
    "'Fetch Job Description', or paste the description manually."
)

job_tab1, job_tab2, job_tab3 = st.tabs(
    [
        "💼 Job 1",
        "💼 Job 2",
        "💼 Job 3",
    ]
)

with job_tab1:
    job1 = job_input(1)

with job_tab2:
    job2 = job_input(2)

with job_tab3:
    job3 = job_input(3)

st.write("")

analyze_col1, analyze_col2, analyze_col3 = st.columns(
    [1, 1.3, 1]
)

with analyze_col2:
    analyze_button = st.button(
        "✨ Analyze My Skills",
        type="primary",
        width="stretch",
    )


should_scroll = False

if analyze_button:
    if not resume_content.strip():
        st.warning("Please add your resume first.")

    else:
        jobs = []

        if job1.strip():
            jobs.append(("Job 1", job1))

        if job2.strip():
            jobs.append(("Job 2", job2))

        if job3.strip():
            jobs.append(("Job 3", job3))

        if len(jobs) == 0:
            st.warning(
                "Please add at least one job description."
            )

        else:
            status_box = st.empty()

            status_box.info(
                "⏳ Starting your skill analysis..."
            )

            progress = st.progress(
                10,
                text="Reading your resume...",
            )

            total_jobs = len(jobs)

            resume_skills = detect_skills(
                resume_content
            )

            progress.progress(
                30,
                text="Finding skills in your resume...",
            )

            all_job_skills = []

            for job_name, job_description in jobs:
                skills = detect_skills(
                    job_description
                )

                all_job_skills.extend(skills)

            skill_count = Counter(all_job_skills)

            progress.progress(
                50,
                text="Comparing skills with target jobs...",
            )

            model = load_model()

            progress.progress(
                65,
                text="Running AI semantic matching...",
            )

            match_results = []

            for index, (
                job_name,
                job_description,
            ) in enumerate(jobs):
                score = calculate_similarity(
                    model,
                    resume_content,
                    job_description,
                )

                match_results.append(
                    (job_name, score)
                )

                progress_value = (
                    65
                    + int(
                        ((index + 1) / total_jobs)
                        * 20
                    )
                )

                progress.progress(
                    progress_value,
                    text=f"Analyzing {job_name}...",
                )

            missing_skills = []

            for skill, count in skill_count.most_common():
                if skill not in resume_skills:
                    missing_skills.append(
                        (skill, count)
                    )

            progress.progress(
                95,
                text="Creating your learning roadmap...",
            )

            average_score = round(
                sum(
                    score
                    for _, score in match_results
                )
                / total_jobs,
                1,
            )

            st.session_state.analysis_results = {
                "total_jobs": total_jobs,
                "resume_skills": resume_skills,
                "skill_count": dict(skill_count),
                "match_results": match_results,
                "average_score": average_score,
                "missing_skills": missing_skills,
            }

            progress.progress(
                100,
                text="Analysis complete!",
            )

            status_box.success(
                "✅ Analysis complete! "
                "Taking you to your results..."
            )

            should_scroll = True


results = st.session_state.analysis_results

if results is not None:
    total_jobs = results["total_jobs"]
    resume_skills = results["resume_skills"]

    skill_count = Counter(
        results["skill_count"]
    )

    match_results = results["match_results"]
    average_score = results["average_score"]
    missing_skills = results["missing_skills"]

    st.divider()

    st.html(
        """
        <div id="analysis-results"></div>
        """
    )

    st.html(
        """
        <div class="analysis-banner">
            <div class="analysis-banner-title">
                📊 Your Skill Analysis
            </div>

            <div class="analysis-banner-text">
                Here's how your resume compares
                with your selected target jobs.
            </div>
        </div>
        """
    )

    metric_columns = st.columns(
        total_jobs + 1
    )

    for index, (
        job_name,
        score,
    ) in enumerate(match_results):
        with metric_columns[index]:
            st.metric(
                job_name,
                f"{score}%",
            )

    with metric_columns[-1]:
        st.metric(
            "Average Match",
            f"{average_score}%",
        )

    st.caption(
        "AI Match represents semantic similarity "
        "between your resume and the selected "
        "job descriptions. It is not your "
        "probability of getting hired."
    )

    st.write("")

    left_result, right_result = st.columns(
        [0.55, 0.45],
        gap="large",
    )

    with left_result:
        with st.container(border=True):
            st.markdown(
                "### 🎯 Skills Found in Target Jobs"
            )

            result_data = []

            for skill, count in skill_count.most_common():
                if skill in resume_skills:
                    status = "✅ Yes"
                else:
                    status = "❌ No"

                result_data.append(
                    {
                        "Skill": skill,
                        "Required": f"{count}/{total_jobs}",
                        "In Resume": status,
                    }
                )

            if result_data:
                result_table = pd.DataFrame(
                    result_data
                )

                show_table(
                    result_table,
                    height=420,
                )
            else:
                st.info(
                    "No known skills were detected "
                    "in the target jobs."
                )

    with right_result:
        with st.container(border=True):
            st.markdown(
                "### ✅ Your Detected Skills"
            )

            if resume_skills:
                st.write(
                    " • ".join(resume_skills)
                )
            else:
                st.info(
                    "No matching technical "
                    "skills were detected."
                )

        with st.container(border=True):
            st.markdown(
                "### 📌 Skills You May Need"
            )

            if missing_skills:
                missing_table_data = []

                for skill, count in missing_skills:
                    priority = get_priority(
                        count,
                        total_jobs,
                    )

                    if priority == "High":
                        priority_display = "🔴 High"

                    elif priority == "Medium":
                        priority_display = "🟠 Medium"

                    else:
                        priority_display = "🟢 Low"

                    missing_table_data.append(
                        {
                            "Skill": skill,
                            "Jobs": f"{count}/{total_jobs}",
                            "Priority": priority_display,
                        }
                    )

                missing_table = pd.DataFrame(
                    missing_table_data
                )

                show_table(
                    missing_table,
                    height=280,
                )

            else:
                st.success(
                    "You already have all detected "
                    "skills required by these jobs."
                )

    if missing_skills:
        st.write("")

        st.html(
            """
            <div class="section-title">
                🗺️ Suggested Learning Roadmap
            </div>

            <div class="section-description">
                Start with the most frequently required
                skills. Click a skill to view the
                learning steps.
            </div>
            """
        )

        roadmap_col1, roadmap_col2 = st.columns(
            2,
            gap="large",
        )

        for index, (
            skill,
            count,
        ) in enumerate(missing_skills):
            priority = get_priority(
                count,
                total_jobs,
            )

            learning_steps = LEARNING_GUIDE.get(
                skill,
                [
                    f"Learn the basics of {skill}",
                    f"Practice {skill}",
                    f"Build a small project using {skill}",
                ],
            )

            if index % 2 == 0:
                target_column = roadmap_col1
            else:
                target_column = roadmap_col2

            with target_column:
                with st.expander(
                    f"{skill}  •  {priority} Priority"
                ):
                    st.caption(
                        f"Required in {count} of "
                        f"{total_jobs} selected job(s)"
                    )

                    for step_number, step in enumerate(
                        learning_steps,
                        start=1,
                    ):
                        st.write(
                            f"**{step_number}.** {step}"
                        )


if should_scroll:
    components.html(
        """
        <script>
        setTimeout(function() {
            const target =
                window.parent.document
                .getElementById(
                    "analysis-results"
                );

            if (target) {
                target.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }
        }, 400);
        </script>
        """,
        height=0,
    )