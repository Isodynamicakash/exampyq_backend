"""
models/schemas.py — Pydantic models for the admin pipeline API
"""

from pydantic import BaseModel
from typing import Optional


class JobStatus(BaseModel):
    job_id:         str
    status:         str            # pending | processing | parsing | ready | failed
    filename:       str
    progress:       int            # 0-100
    error:          Optional[str] = None
    question_count: Optional[int] = None
    image_count:    Optional[int] = None


class ParsedQuestionIn(BaseModel):
    number:         int            = 0
    q_type:         str            = "MCQ"     # MCQ | MSQ | NUMERICAL
    subject:        str            = ""
    section:        str            = ""
    year:           Optional[year]  = None
    shift:          str            = ""
    exam_name:      str            = "JEE Main"
    exam_date:      Optional[str]  = None
    question:       str            = ""
    options:        list[str]      = []
    answer:         str            = ""
    solution:       str            = ""
    q_images:       list[str]      = []
    sol_images:     list[str]      = []
    opt_images:     dict           = {}        # {a: img_id, b: img_id, c: img_id, d: img_id}
    images_found:   list[str]      = []
    images_missing: list[str]      = []

    # admin-filled / LLM-tagged
    paper_id:       Optional[int]  = None
    chapter_id:     Optional[int]  = None
    topic_id:       Optional[int]  = None
    chapter_name:   str            = ""
    topic_name:     str            = ""
    topic:          str            = ""        # legacy alias
    subject_name:   str            = ""        # explicit subject name (for display/edit)
    difficulty:     Optional[str]  = None      # easy | medium | hard
    marks_correct:  int            = 4
    marks_wrong:    int            = -1
    verified:       bool           = False
    _isManual:      bool           = False
    _manualId:      Optional[str]  = None


class SaveQuestionsRequest(BaseModel):
    job_id:    str
    questions: list[ParsedQuestionIn]


class SaveQuestionsResponse(BaseModel):
    saved_count:  int
    question_ids: list[int]