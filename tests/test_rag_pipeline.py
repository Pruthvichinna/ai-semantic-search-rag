"""
Basic smoke tests so the repo looks professional.
Run:
    pytest -q
"""

from pathlib import Path

def test_project_structure():
    expected = [
        "README.md",
        "requirements.txt",
        "config/config.example.json",
        "code/index_documents.py",
        "code/rag_pipeline.py",
        "code/prompt_templates.py",
        "code/monitoring_dashboard.py",
        "data/sample_docs",
        "data/embeddings",
    ]
    for p in expected:
        assert Path(p).exists(), f"Missing: {p}"
