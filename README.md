# Job Skill Helper
Job Skill Helper is an AI-powered resume-to-job skill-gap analyzer developed as part of the IBM SkillsBuild - AICTE - BharatCares Machine Learning & Applied AI Internship Program 2026.
The project is aligned with **SDG 8 - Decent Work and Economic Growth**.

## Problem
Students, fresh graduates, and early-career professionals often find it difficult to understand which technical skills they are missing for their target jobs.
Different job descriptions may require different skills, making it difficult to decide what to learn first.

## Solution
Job Skill Helper allows users to:
- Upload a resume PDF or paste resume text
- Add up to three target jobs
- Fetch a job description from a public job URL
- Paste job descriptions manually
- Detect technical skills from the resume and job descriptions
- Compare the resume with each job using semantic similarity
- Identify missing skills
- Prioritize missing skills based on how many selected jobs require them
- View a suggested learning roadmap

## How Machine Learning Is Used
The application uses the pretrained Sentence Transformer model:
`all-MiniLM-L6-v2`

The model converts the resume and job descriptions into numerical embeddings.
Cosine similarity is then used to measure the semantic similarity between the resume and each selected job description.
Skill detection is performed using a predefined skill list and aliases.
The AI Match score represents semantic similarity only. It does not represent the probability of getting hired.

## Tech Stack
- Python
- Streamlit
- Sentence Transformers
- Scikit-learn
- PyMuPDF
- Pandas
- BeautifulSoup
- Requests

## Run the Project Locally
Clone the repository:

```bash
git clone https://github.com/Mitanshu-Singhal/job-skill-helper.git
```

Go to the project folder:

```bash
cd job-skill-helper
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

The application will normally open at:

`http://localhost:8501`
