"""Fills the four collections with PLACEHOLDER passages so the agent can be developed
and tested before the real data exists. All dummy rows have dummy=True so they can be wiped.
Usage: python seed_dummy_data.py [--reset]"""
import sys
from careergpt.config import COLLECTIONS
from careergpt.store import get_collection, upsert_passages

DUMMY = {
    "blogs": [
        ("ExampleCorp runs analytics internship hiring on campus; roles: Data Analyst Intern, stipend 80k/month (dummy).", "placement_blog_2025"),
        ("Shortlisting for internships typically begins in the first week of October, followed by an aptitude test then interviews (dummy).", "internship_blog_2025"),
        ("FinCo visits for quant and finance roles; process: online test, two technical interviews, HR (dummy).", "placement_blog_2024"),
    ],
    "booklets": [
        ("For analytics roles, prepare SQL, statistics, Python and case-style questions (dummy).", "internship_guide_101"),
        ("Keep resumes to one page, lead with projects, quantify impact (dummy).", "placement_blue_book"),
        ("For consulting, practice structured case solving and guesstimates (dummy).", "placement_blue_book"),
    ],
    "resumes": [
        ("Data-role resumes commonly feature ML projects with real datasets, Kaggle competitions, dashboards (dummy pattern).", "resume_repo"),
        ("Consulting-aligned PORs: club coordinator, case competition lead, E-Cell roles (dummy pattern).", "resume_repo"),
    ],
    "policies": [
        ("Placement registration requires completing the profile on the portal before the deadline (dummy).", "institute_policy_page"),
        ("Resume verification is done by the department coordinator before the first slot (dummy).", "institute_policy_page"),
    ],
}

if __name__ == "__main__":
    if "--reset" in sys.argv:
        for key in COLLECTIONS:
            col = get_collection(key)
            ids = col.get(where={"dummy": True})["ids"]
            if ids:
                col.delete(ids=ids)
        print("Dummy rows removed.")
    else:
        for key, rows in DUMMY.items():
            upsert_passages(key, [
                {"id": f"dummy-{key}-{i}", "text": t, "source": s, "dummy": True}
                for i, (t, s) in enumerate(rows)
            ])
        print("Seeded dummy data into:", ", ".join(COLLECTIONS.values()))
