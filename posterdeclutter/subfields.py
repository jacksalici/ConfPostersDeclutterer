"""Assign a subfield to each poster.

Preference order:
  1. the arXiv primary category, when the paper was found  (authoritative)
  2. a keyword rule-book over the OCR text                 (deterministic)
  3. "Unclassified"                                        (honest)

An LLM is never required here; see llm.py for the opt-in third pass.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .util import normalise

# arXiv category -> human readable subfield. Exact ids first, then the archive
# prefix, then the top-level group.
CATEGORY_NAMES: Dict[str, str] = {
    "astro-ph.CO": "Cosmology & Extragalactic Astrophysics",
    "astro-ph.GA": "Galactic Astrophysics",
    "astro-ph.HE": "High Energy Astrophysics",
    "astro-ph.IM": "Astronomical Instrumentation",
    "astro-ph.SR": "Solar & Stellar Astrophysics",
    "astro-ph.EP": "Earth & Planetary Astrophysics",
    "cond-mat.mes-hall": "Mesoscale & Nanoscale Physics",
    "cond-mat.mtrl-sci": "Materials Science",
    "cond-mat.stat-mech": "Statistical Mechanics",
    "cond-mat.str-el": "Strongly Correlated Electrons",
    "cond-mat.supr-con": "Superconductivity",
    "cond-mat.soft": "Soft Condensed Matter",
    "cond-mat.quant-gas": "Quantum Gases",
    "cond-mat.dis-nn": "Disordered Systems & Neural Networks",
    "cs.AI": "Artificial Intelligence",
    "cs.AR": "Hardware Architecture",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Computational Science & Engineering",
    "cs.CG": "Computational Geometry",
    "cs.CL": "Natural Language Processing",
    "cs.CR": "Security & Cryptography",
    "cs.CV": "Computer Vision",
    "cs.CY": "Computers & Society",
    "cs.DB": "Databases",
    "cs.DC": "Distributed & Parallel Computing",
    "cs.DS": "Data Structures & Algorithms",
    "cs.GT": "Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.LG": "Machine Learning",
    "cs.LO": "Logic in Computer Science",
    "cs.MA": "Multiagent Systems",
    "cs.NE": "Neural & Evolutionary Computing",
    "cs.NI": "Networking",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SD": "Sound & Audio",
    "cs.SE": "Software Engineering",
    "cs.SI": "Social & Information Networks",
    "eess.AS": "Audio & Speech Processing",
    "eess.IV": "Image & Video Processing",
    "eess.SP": "Signal Processing",
    "eess.SY": "Systems & Control",
    "gr-qc": "General Relativity & Quantum Cosmology",
    "hep-ex": "High Energy Physics - Experiment",
    "hep-lat": "Lattice Field Theory",
    "hep-ph": "High Energy Physics - Phenomenology",
    "hep-th": "High Energy Physics - Theory",
    "math-ph": "Mathematical Physics",
    "nucl-ex": "Nuclear Experiment",
    "nucl-th": "Nuclear Theory",
    "physics.acc-ph": "Accelerator Physics",
    "physics.app-ph": "Applied Physics",
    "physics.atom-ph": "Atomic Physics",
    "physics.bio-ph": "Biological Physics",
    "physics.chem-ph": "Chemical Physics",
    "physics.comp-ph": "Computational Physics",
    "physics.data-an": "Data Analysis & Statistics",
    "physics.flu-dyn": "Fluid Dynamics",
    "physics.ins-det": "Instrumentation & Detectors",
    "physics.med-ph": "Medical Physics",
    "physics.optics": "Optics",
    "physics.plasm-ph": "Plasma Physics",
    "physics.space-ph": "Space Physics",
    "q-bio.BM": "Biomolecules",
    "q-bio.GN": "Genomics",
    "q-bio.NC": "Neuroscience",
    "q-bio.PE": "Populations & Evolution",
    "q-bio.QM": "Quantitative Methods (Biology)",
    "q-fin.ST": "Statistical Finance",
    "quant-ph": "Quantum Physics",
    "stat.AP": "Applied Statistics",
    "stat.ME": "Statistical Methodology",
    "stat.ML": "Machine Learning (Statistics)",
}

ARCHIVE_NAMES: Dict[str, str] = {
    "astro-ph": "Astrophysics",
    "cond-mat": "Condensed Matter",
    "cs": "Computer Science",
    "econ": "Economics",
    "eess": "Electrical Engineering & Systems Science",
    "math": "Mathematics",
    "nlin": "Nonlinear Sciences",
    "physics": "Physics (other)",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "stat": "Statistics",
}

UNCLASSIFIED = "Unclassified"

# Keyword rule-book: (subfield, weight-1 terms). Matched on normalised text as
# whole words; the subfield with the most distinct hits wins.
KEYWORD_RULES: List[Tuple[str, List[str]]] = [
    ("Machine Learning", [
        "neural network", "deep learning", "transformer", "gradient descent",
        "training set", "fine tuning", "self supervised", "diffusion model",
        "graph neural", "attention", "benchmark accuracy", "overfitting",
    ]),
    ("Natural Language Processing", [
        "language model", "llm", "tokenizer", "text corpus", "question answering",
        "machine translation", "named entity", "prompt", "summarization",
    ]),
    ("Computer Vision", [
        "image segmentation", "object detection", "convolutional", "image classification",
        "point cloud", "semantic segmentation", "optical flow", "pose estimation",
    ]),
    ("Robotics", [
        "manipulator", "grasping", "slam", "motion planning", "quadruped",
        "teleoperation", "end effector", "reinforcement learning policy",
    ]),
    ("High Energy Physics - Experiment", [
        "lhc", "atlas", "cms", "lhcb", "collider", "luminosity", "jet tagging",
        "cross section measurement", "trigger efficiency", "pileup",
    ]),
    ("High Energy Physics - Phenomenology", [
        "beyond the standard model", "dark matter candidate", "supersymmetry",
        "effective field theory", "parton distribution", "neutrino mass",
    ]),
    ("Astrophysics", [
        "galaxy", "galaxies", "supernova", "exoplanet", "redshift", "telescope",
        "stellar", "cosmic ray", "black hole", "gravitational wave", "pulsar",
    ]),
    ("Quantum Physics", [
        "qubit", "entanglement", "quantum circuit", "decoherence", "quantum error",
        "superposition", "quantum computer", "bell state",
    ]),
    ("Condensed Matter", [
        "superconductor", "superconducting gap", "spin liquid", "graphene",
        "lattice model", "phase transition", "magnetization", "band structure",
    ]),
    ("Instrumentation & Detectors", [
        "calorimeter", "silicon detector", "photomultiplier", "readout electronics",
        "scintillator", "test beam", "cryostat", "data acquisition",
    ]),
    ("Neuroscience", [
        "eeg", "fmri", "cortex", "neuron spike", "connectome", "synaptic",
        "brain activity", "electrophysiology",
    ]),
    ("Biomolecules", [
        "protein structure", "molecular dynamics", "binding affinity", "rna",
        "genome", "docking", "enzyme", "crystallography",
    ]),
    ("Medical Physics", [
        "radiotherapy", "dosimetry", "ct scan", "mri", "tumour", "tumor",
        "clinical trial", "patient cohort",
    ]),
    ("Climate & Earth Science", [
        "climate model", "atmospheric", "ocean", "precipitation", "emissions",
        "remote sensing", "sea ice",
    ]),
    ("Statistics", [
        "bayesian", "posterior distribution", "hypothesis test", "confidence interval",
        "markov chain monte carlo", "estimator", "p value",
    ]),
]


def name_for_category(category: str) -> str:
    if not category:
        return UNCLASSIFIED
    if category in CATEGORY_NAMES:
        return CATEGORY_NAMES[category]
    archive = category.split(".", 1)[0]
    if archive in ARCHIVE_NAMES:
        return ARCHIVE_NAMES[archive]
    return category


def classify_by_keywords(text: str) -> Tuple[Optional[str], float, List[str]]:
    """Return (subfield, confidence 0..1, matched terms)."""
    haystack = " " + normalise(text) + " "
    scores = []
    for subfield, terms in KEYWORD_RULES:
        hits = [t for t in terms if re.search(r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(t), haystack)]
        if hits:
            scores.append((len(hits), subfield, hits))
    if not scores:
        return None, 0.0, []
    scores.sort(reverse=True)
    top_hits, subfield, hits = scores[0]
    runner_up = scores[1][0] if len(scores) > 1 else 0
    margin = (top_hits - runner_up) / float(top_hits)
    confidence = min(1.0, 0.35 + 0.2 * top_hits) * (0.6 + 0.4 * margin)
    return subfield, round(confidence, 3), hits


def classify(primary_category: str, text: str, subject: str = "",
             subject_confidence: float = 0.0) -> Tuple[str, str, float, List[str]]:
    """Return (subfield, source, confidence, evidence).

    Preference order, most trustworthy first:
      1. the arXiv primary category - declared by the authors themselves
      2. the topic the matched paper carries (OpenAlex/Crossref) - assigned by
         the index, but attached to a paper we matched above threshold
      3. keyword rules over the OCR text - no paper needed, but OCR-noisy
      4. "Unclassified"
    """
    if primary_category:
        return name_for_category(primary_category), "arxiv", 1.0, [primary_category]
    if subject:
        return subject, "index", round(subject_confidence or 0.6, 3), [subject]
    guess, confidence, hits = classify_by_keywords(text)
    if guess:
        return guess, "keywords", confidence, hits
    return UNCLASSIFIED, "none", 0.0, []
