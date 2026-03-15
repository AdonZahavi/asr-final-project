from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from common_voice import ensure_parent_dir, tsv_dict_reader


HEBREW_DIACRITICS_RE = re.compile(r"[\u0591-\u05C7]")
HEBREW_TOKEN_RE = re.compile(r"[^\d\u05d0-\u05ea]")
PUNCT_TRANSLATION = str.maketrans(
    {
        "\"": " ",
        "'": " ",
        ",": " ",
        ".": " ",
        ":": " ",
        ";": " ",
        "!": " ",
        "?": " ",
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
        "{": " ",
        "}": " ",
        "<": " ",
        ">": " ",
        "_": " ",
        "-": " ",
        "־": " ",
        "–": " ",
        "—": " ",
        "/": " ",
        "\\": " ",
    }
)

ORTHOGRAPHIC_VARIANTS = {
    "\u05d4\u05d9\u05d9\u05ea\u05d4": "\u05d4\u05d9\u05ea\u05d4",
    "\u05d4\u05db\u05d5\u05dc": "\u05d4\u05db\u05dc",
    "\u05de\u05d9\u05d9\u05d3": "\u05de\u05d9\u05d3",
    "\u05de\u05d9\u05e1\u05d9\u05dd": "\u05de\u05e1\u05d9\u05dd",
    "\u05e9\u05e0\u05d9\u05d9\u05d4": "\u05e9\u05e0\u05d9\u05d4",
    "\u05d0\u05de\u05d0": "\u05d0\u05d9\u05de\u05d0",
    "\u05de\u05d0\u05d3": "\u05de\u05d0\u05d5\u05d3",
    "\u05de\u05e8\u05e6": "\u05de\u05e8\u05e5",
    "\u05d0\u05dc\u05d9": "\u05d0\u05dc\u05d9\u05d9",
    "\u05d0\u05ea\u05d9": "\u05d0\u05d9\u05ea\u05d9",
    "\u05d5\u05d0\u05dc\u05d5": "\u05d5\u05d0\u05d9\u05dc\u05d5",
    "\u05d1\u05d0\u05d6\u05e0\u05d9\u05da": "\u05d1\u05d0\u05d5\u05d6\u05e0\u05d9\u05da",
    "\u05e7\u05d5\u05e7\u05d8\u05d9\u05dc": "\u05e7\u05d5\u05e7\u05d8\u05d9\u05d9\u05dc",
    "\u05d1\u05d9\u05d3\u05d9\u05dd": "\u05d1\u05d9\u05d3\u05d9\u05d9\u05dd",
    "\u05e8\u05d5\u05d5\u05d9": "\u05e8\u05d1\u05d5\u05d9",
}

PARTNER_ADDITIONAL_REPLACEMENTS = {
    "\u05d4\u05d9\u05ea\u05d4": "\u05d4\u05d9\u05d9\u05ea\u05d4",
    "\u05d0\u05d9\u05ea\u05d9": "\u05d0\u05ea\u05d9",
    "\u05de\u05e1\u05d9\u05dd": "\u05de\u05d9\u05e1\u05d9\u05dd",
    "\u05d4\u05d1\u05d9\u05df \u05dc\u05d0\u05d5\u05de\u05d9\u05d9\u05dd": "\u05d4\u05d1\u05d9\u05e0\u05dc\u05d0\u05d5\u05de\u05d9\u05d9\u05dd",
    "\u05d1\u05d9\u05ea \u05ea\u05e4\u05d9\u05dc\u05d4": "\u05d1\u05d9\u05ea \u05ea\u05e4\u05d9\u05dc\u05d4",
    "\u05d5\u05d4\u05d0\u05de\u05e0\u05d5\u05ea": "\u05d5\u05d4\u05d0\u05d5\u05de\u05e0\u05d5\u05ea",
    "\u05d0\u05dc\u05d9\u05d9": "\u05d0\u05dc\u05d9",
    "\u05d9\u05d9\u05e4\u05dc\u05d5": "\u05d9\u05e4\u05dc\u05d5",
    "\u05d1\u05e1\u05e4\u05d5\u05d8\u05d9\u05e4\u05d9\u05d9": "\u05d1\u05e1\u05e4\u05d5\u05d8\u05d9\u05e4\u05d9",
    "\u05e9\u05e9\u05d9\u05d9\u05da": "\u05e9\u05e9\u05d9\u05da",
    "\u05dc\u05e2\u05e1\u05ea": "\u05dc\u05e2\u05d9\u05e1\u05ea",
    "\u05d0\u05e9\u05d4": "\u05d0\u05d9\u05e9\u05d4",
    "\u05d0\u05dc\u05d4\u05d9\u05d4\u05dd": "\u05d0\u05dc\u05d5\u05d4\u05d9\u05d4\u05dd",
    "\u05d5\u05de\u05ea\u05db\u05d5\u05e6\u05d9\u05dd": "\u05d5\u05de\u05ea\u05db\u05d5\u05d5\u05e6\u05d9\u05dd",
    "\u05db\u05dc\u05dd": "\u05db\u05d5\u05dc\u05dd",
    "\u05dc\u05e2\u05ea\u05d9\u05dd": "\u05dc\u05e2\u05d9\u05ea\u05d9\u05dd",
    "\u05d2\u05dc\u05d9\u05e0\u05d5": "\u05d2\u05d9\u05dc\u05d9\u05e0\u05d5",
    "\u05db\u05e9\u05d1\u05d5\u05e2\u05d9\u05dd": "\u05db\u05e9\u05d1\u05d5\u05e2\u05d9\u05d9\u05dd",
    "\u05d1\u05de\u05d3\u05d4": "\u05d1\u05de\u05d9\u05d3\u05d4",
    "\u05d0\u05d9\u05de\u05d4": "\u05d0\u05d9\u05d5\u05de\u05d4",
    "\u05de\u05dc\u05db\u05d3\u05ea": "\u05de\u05dc\u05db\u05d5\u05d3\u05ea",
    "\u05d0\u05e4\u05dc\u05d5": "\u05d0\u05e4\u05d9\u05dc\u05d5",
    "\u05de\u05d8\u05ea\u05d4": "\u05de\u05d9\u05d8\u05ea\u05d4",
    "\u05d1\u05d0\u05d5\u05e4\u05dc": "\u05d1\u05d0\u05e4\u05dc",
    "\u05e2\u05db\u05e9\u05d5": "\u05e2\u05db\u05e9\u05d9\u05d5",
    "\u05ea\u05d7\u05d6\u05e8": "\u05ea\u05d7\u05d6\u05d5\u05e8",
    "\u05dc\u05d3\u05d5\u05d2\u05de\u05d4": "\u05dc\u05d3\u05d5\u05d2\u05de\u05d0",
    "\u05d2\u05d9\u05d4\u05d9\u05e0\u05d5\u05dd": "\u05d2\u05d9\u05d4\u05e0\u05d5\u05dd",
    "\u05de\u05d9\u05e0\u05d4\u05dc\u05d9": "\u05de\u05e0\u05d4\u05dc\u05d9",
    "\u05d2\u05d6\u05d9\u05e8\u05d5\u05ea": "\u05d2\u05d6\u05e8\u05d5\u05ea",
    "\u05dc\u05d9\u05e6\u05d2": "\u05dc\u05d9\u05d9\u05e6\u05d2",
    "\u05e4\u05d9\u05e1\u05d1\u05d5\u05e7": "\u05e4\u05d9\u05d9\u05e1\u05d1\u05d5\u05e7",
    "\u05d0\u05dc\u05d8\u05e8\u05e0\u05d8\u05d9\u05d1\u05d9\u05dd": "\u05d0\u05dc\u05d8\u05e8\u05e0\u05d8\u05d9\u05d1\u05d9\u05d9\u05dd",
    "\u05d4\u05e8\u05d9\u05d9\u05e0\u05d5": "\u05d4\u05e8\u05d9\u05e0\u05d5",
    "\u05dc\u05e2\u05d9\u05e4\u05d4": "\u05dc\u05e2\u05d9\u05d9\u05e4\u05d4",
    "\u05db\u05d9\u05e9\u05e8\u05d5\u05e0\u05d5\u05ea": "\u05db\u05e9\u05e8\u05d5\u05e0\u05d5\u05ea",
    "\u05d4\u05d6\u05d9\u05e7\u05e0\u05d4": "\u05d4\u05d6\u05e7\u05e0\u05d4",
    "\u05d0\u05d4\u05e8\u05df": "\u05d0\u05d4\u05e8\u05d5\u05df",
    "\u05d1\u05d3\u05d1\u05e8\u05d9": "\u05d1\u05d3\u05d1\u05e8\u05d9\u05d9",
    "\u05d4\u05de\u05dc\u05d9\u05d5\u05e0\u05d9\u05dd": "\u05d4\u05de\u05d9\u05dc\u05d9\u05d5\u05e0\u05d9\u05dd",
    "\u05d1\u05d4\u05e2\u05d3\u05e8": "\u05d1\u05d4\u05d9\u05e2\u05d3\u05e8",
    "\u05d4\u05ea\u05d9\u05e9\u05d1\u05d5\u05ea": "\u05d4\u05ea\u05d9\u05d9\u05e9\u05d1\u05d5\u05ea",
    "\u05d4\u05d9\u05e2\u05d3\u05e8": "\u05d4\u05e2\u05d3\u05e8",
    "\u05dc\u05d9\u05d1\u05d9": "\u05dc\u05d1\u05d9",
    "\u05de\u05e6\u05d3\u05dd": "\u05de\u05e6\u05d9\u05d3\u05dd",
    "\u05e6\u05e4\u05d5\u05e8\u05d4": "\u05e6\u05d9\u05e4\u05d5\u05e8\u05d4",
    "\u05ea\u05e1\u05e2": "\u05ea\u05d9\u05e1\u05e2",
    "\u05d6\u05d9\u05db\u05e8\u05d5\u05e0\u05d5\u05ea": "\u05d6\u05db\u05e8\u05d5\u05e0\u05d5\u05ea",
    "\u05e0\u05d4\u05e8\u05d9\u05d9\u05d4": "\u05e0\u05d4\u05e8\u05d9\u05d4",
    "\u05d9\u05e7\u05d7 ": "\u05d9\u05d9\u05e7\u05d7 ",
    "\u05dc\u05e7\u05e1\u05e7\u05dc\u05d9": "\u05dc\u05e7\u05e1\u05d9\u05e7\u05dc\u05d9",
    "\u05d5\u05de\u05db\u05e2\u05e8": "\u05d5\u05de\u05db\u05d5\u05e2\u05e8",
    "\u05d4\u05de\u05d9\u05e0\u05d4\u05dc\u05d9\u05d5\u05ea": "\u05d4\u05de\u05e0\u05d4\u05dc\u05d9\u05d5\u05ea",
    "\u05dc\u05d0\u05e4\u05e0\u05d5": "\u05dc\u05d0\u05e4\u05d9\u05e0\u05d5",
    " \u05e8\u05d1\u05d5\u05ea\u05d9": " \u05e8\u05d1\u05d5\u05ea\u05d9\u05d9",
    "\u05d4\u05de\u05dc\u05d9\u05dd": "\u05d4\u05de\u05d9\u05dc\u05d9\u05dd",
    "\u05d0\u05d9\u05e0\u05ea\u05d9\u05e4\u05d3\u05ea": "\u05d0\u05d9\u05e0\u05ea\u05d9\u05e4\u05d0\u05d3\u05ea",
    "\u05e9\u05ea\u05d9\u05dd": "\u05e9\u05ea\u05d9\u05d9\u05dd",
    " \u05d3\u05e8 ": " \u05d3\u05d5\u05e7\u05d8\u05d5\u05e8 ",
    " \u05d1 ": "\u05d1 ",
    " \u05dc ": "\u05dc ",
    " \u05de ": "\u05de ",
    " \u05db ": "\u05db ",
    "\u05d1\u05e9\u05e2\u05d4 \u05d0\u05e8\u05d1\u05e2": "\u05d1\u05e9\u05e2\u05d4 16",
    "\u05e9\u05d1\u05e2\u05d9\u05dd \u05d0\u05d7\u05d5\u05d6\u05d9\u05dd": "70%",
    "\u05d7\u05de\u05d9\u05e9\u05d4 \u05d0\u05d7\u05d5\u05d6": "5%",
    "\u05e9\u05de\u05d5\u05e0\u05d9\u05dd \u05d0\u05d7\u05d5\u05d6\u05d9\u05dd": "80%",
    "\u05e2\u05e9\u05e8\u05d9\u05dd \u05d5\u05ea\u05e9\u05e2\u05d4 \u05d0\u05d7\u05d5\u05d6\u05d9\u05dd": "29%",
    "\u05e2\u05e9\u05e8\u05d9\u05dd \u05d5\u05e9\u05d9\u05e9\u05d4": "26",
    "\u05d0\u05e8\u05d1\u05e2 \u05e2\u05e9\u05e8\u05d4": "14",
    "\u05db\u05d0\u05e8\u05d1\u05e2 \u05de\u05d0\u05d5\u05ea \u05d0\u05dc\u05e3": "400000\u05db",
    "\u05de\u05d0\u05ea\u05d9\u05d9\u05dd \u05e9\u05de\u05d5\u05e0\u05d9\u05dd \u05d5\u05d0\u05d7\u05d3 \u05d0\u05dc\u05e3": "281000",
    "\u05de\u05d0\u05d4 \u05d5\u05d7\u05de\u05d9\u05e9\u05d9\u05dd \u05d0\u05dc\u05e3": "150000",
    "\u05e9\u05dc\u05d5\u05e9\u05d9\u05dd \u05d0\u05dc\u05e3": "30000",
    "\u05e2\u05e9\u05e8\u05d9\u05dd \u05d5\u05e9\u05dc\u05d5\u05e9\u05d4 \u05d0\u05dc\u05e3": "23000",
    "\u05e2\u05e9\u05e8\u05d9\u05dd \u05d5\u05e9\u05e0\u05d9\u05d9\u05dd \u05d0\u05dc\u05e3": "22000",
    "\u05d7\u05de\u05e9\u05ea \u05d0\u05dc\u05e4\u05d9\u05dd": "5000",
    "\u05d0\u05dc\u05e4\u05d9\u05d9\u05dd \u05d5\u05d0\u05d7\u05ea \u05e2\u05e9\u05e8\u05d4": "2011",
    "\u05d0\u05dc\u05e4\u05d9\u05d9\u05dd \u05d5\u05e9\u05dc\u05d5\u05e9 \u05e2\u05e9\u05e8\u05d4": "2013",
    "\u05d0\u05dc\u05e4\u05d9\u05d9\u05dd \u05d5\u05e9\u05ea\u05d9\u05dd \u05e2\u05e9\u05e8\u05d4": "2012",
    "\u05d0\u05dc\u05e4\u05d9\u05d9\u05dd \u05d5\u05e9\u05de\u05d5\u05e0\u05d4": "2008",
    "\u05d0\u05dc\u05e4\u05d9\u05d9\u05dd \u05d5\u05e9\u05e9": "2006",
    "\u05d0\u05dc\u05e4\u05d9\u05d9\u05dd \u05d5\u05e9\u05dc\u05d5\u05e9": "2003",
    "\u05d0\u05dc\u05e4\u05d9\u05d9\u05dd": "2000",
    "\u05d0\u05dc\u05e3 \u05de\u05d0\u05d4 \u05d5\u05e9\u05de\u05d5\u05e0\u05d9\u05dd": "1180",
    "\u05d0\u05dc\u05e3 \u05d5\u05d7\u05de\u05e9 \u05de\u05d0\u05d5\u05ea": "1500",
    "\u05d0\u05dc\u05e3 \u05ea\u05e9\u05e2 \u05de\u05d0\u05d5\u05ea \u05e9\u05de\u05d5\u05e0\u05d9\u05dd \u05d5\u05e9\u05de\u05d5\u05e0\u05d4": "1988",
    "\u05d0\u05dc\u05e3 \u05ea\u05e9\u05e2 \u05de\u05d0\u05d5\u05ea \u05d7\u05de\u05d9\u05e9\u05d9\u05dd \u05d5\u05e9\u05e9": "1956",
    "\u05de\u05d0\u05d4 \u05d5\u05d0\u05e8\u05d1\u05e2\u05d9\u05dd": "140",
    "\u05e2\u05e9\u05e8\u05d9\u05dd \u05d5\u05ea\u05e9\u05e2\u05d4": "29",
    "\u05e2\u05e9\u05e8\u05d9\u05dd": "20",
    "\u05e9\u05ea\u05d9\u05dd \u05e2\u05e9\u05e8\u05d4": "12",
    "\u05d7\u05de\u05d9\u05e9\u05d4 \u05e2\u05e9\u05e8": "15",
    "\u05e9\u05dc\u05d5\u05e9 \u05e2\u05e9\u05e8\u05d4": "13",
    "\u05e9\u05d1\u05e2\u05d4": "7",
    "\u05e9\u05d1\u05e2": "7",
}

SINGLE_NUMBER_WORDS = {
    "\u05d0\u05e4\u05e1": "0",
    "\u05d0\u05d7\u05d3": "1",
    "\u05d0\u05d7\u05ea": "1",
    "\u05e9\u05ea\u05d9\u05d9\u05dd": "2",
    "\u05e9\u05e0\u05d9\u05d9\u05dd": "2",
    "\u05e9\u05ea\u05d9\u05dd": "2",
    "\u05e9\u05e0\u05d9\u05dd": "2",
    "\u05e9\u05dc\u05d5\u05e9": "3",
    "\u05e9\u05dc\u05d5\u05e9\u05d4": "3",
    "\u05d0\u05e8\u05d1\u05e2": "4",
    "\u05d0\u05e8\u05d1\u05e2\u05d4": "4",
    "\u05d7\u05de\u05e9": "5",
    "\u05d7\u05de\u05d9\u05e9\u05d4": "5",
    "\u05e9\u05e9": "6",
    "\u05e9\u05d9\u05e9\u05d4": "6",
    "\u05e9\u05d1\u05e2": "7",
    "\u05e9\u05d1\u05e2\u05d4": "7",
    "\u05e9\u05de\u05d5\u05e0\u05d4": "8",
    "\u05ea\u05e9\u05e2": "9",
    "\u05ea\u05e9\u05e2\u05d4": "9",
    "\u05e2\u05e9\u05e8": "10",
    "\u05e2\u05e9\u05e8\u05d4": "10",
    "\u05e2\u05e9\u05e8\u05d9\u05dd": "20",
    "\u05e9\u05dc\u05d5\u05e9\u05d9\u05dd": "30",
    "\u05d0\u05e8\u05d1\u05e2\u05d9\u05dd": "40",
    "\u05d7\u05de\u05d9\u05e9\u05d9\u05dd": "50",
    "\u05e9\u05d9\u05e9\u05d9\u05dd": "60",
    "\u05e9\u05d1\u05e2\u05d9\u05dd": "70",
    "\u05e9\u05de\u05d5\u05e0\u05d9\u05dd": "80",
    "\u05ea\u05e9\u05e2\u05d9\u05dd": "90",
    "\u05de\u05d0\u05d4": "100",
    "\u05de\u05d0\u05ea\u05d9\u05d9\u05dd": "200",
    "\u05d0\u05dc\u05e3": "1000",
    "\u05d0\u05dc\u05e4\u05d9\u05d9\u05dd": "2000",
}

PHRASE_NUMBER_WORDS = {
    "\u05e9\u05dc\u05d5\u05e9 \u05de\u05d0\u05d5\u05ea": "300",
    "\u05d0\u05e8\u05d1\u05e2 \u05de\u05d0\u05d5\u05ea": "400",
    "\u05d7\u05de\u05e9 \u05de\u05d0\u05d5\u05ea": "500",
    "\u05e9\u05e9 \u05de\u05d0\u05d5\u05ea": "600",
    "\u05e9\u05d1\u05e2 \u05de\u05d0\u05d5\u05ea": "700",
    "\u05e9\u05de\u05d5\u05e0\u05d4 \u05de\u05d0\u05d5\u05ea": "800",
    "\u05ea\u05e9\u05e2 \u05de\u05d0\u05d5\u05ea": "900",
}

NUMBER_PREFIXES = ("\u05d5", "\u05d1", "\u05dc", "\u05db", "\u05de", "\u05d4", "\u05e9")
PERCENT_WORDS = {"\u05d0\u05d7\u05d5\u05d6", "\u05d0\u05d7\u05d5\u05d6\u05d9\u05dd"}


class EditWeights:
    def pair_weight(self, first_obj, second_obj) -> float:
        raise NotImplementedError

    def insertion_weight(self, obj) -> float:
        raise NotImplementedError

    def deletion_weight(self, obj) -> float:
        raise NotImplementedError


class WordEditWeights(EditWeights):
    def pair_weight(self, first_obj, second_obj) -> float:
        return 0.0 if first_obj == second_obj else -1.0

    def insertion_weight(self, obj) -> float:
        return -1.0

    def deletion_weight(self, obj) -> float:
        return -1.0


WordLevelEditWeights = WordEditWeights


def sequences_align(
    first_seq: Iterable[str],
    second_seq: Iterable[str],
    weights: EditWeights,
) -> Tuple[float, List[Tuple[str | None, str | None]]]:
    first_seq = list(first_seq)
    second_seq = list(second_seq)

    _OP_NULL = 0
    _OP_PAIR = 1
    _OP_INS = 2
    _OP_DEL = 3

    first_len = len(first_seq)
    second_len = len(second_seq)
    scores_mat = np.zeros((first_len + 1, second_len + 1), dtype=np.float64)
    ops_mat = np.zeros((first_len + 1, second_len + 1), dtype=np.int8)

    for j, second_obj in enumerate(second_seq, start=1):
        scores_mat[0, j] = scores_mat[0, j - 1] + weights.insertion_weight(second_obj)
        ops_mat[0, j] = _OP_INS

    for i, first_obj in enumerate(first_seq, start=1):
        scores_mat[i, 0] = scores_mat[i - 1, 0] + weights.deletion_weight(first_obj)
        ops_mat[i, 0] = _OP_DEL

        for j, second_obj in enumerate(second_seq, start=1):
            pair_weight = scores_mat[i - 1, j - 1] + weights.pair_weight(first_obj, second_obj)
            insert_weight = scores_mat[i, j - 1] + weights.insertion_weight(second_obj)
            delete_weight = scores_mat[i - 1, j] + weights.deletion_weight(first_obj)

            best_weight = pair_weight
            best_op = _OP_PAIR
            if insert_weight > best_weight:
                best_weight = insert_weight
                best_op = _OP_INS
            if delete_weight > best_weight:
                best_weight = delete_weight
                best_op = _OP_DEL

            scores_mat[i, j] = best_weight
            ops_mat[i, j] = best_op

    aligned_pairs: List[Tuple[str | None, str | None]] = []
    i = first_len
    j = second_len
    while i > 0 or j > 0:
        curr_op = ops_mat[i, j]
        if curr_op == _OP_PAIR:
            i -= 1
            j -= 1
            aligned_pairs.append((first_seq[i], second_seq[j]))
        elif curr_op == _OP_INS:
            j -= 1
            aligned_pairs.append((None, second_seq[j]))
        elif curr_op == _OP_DEL:
            i -= 1
            aligned_pairs.append((first_seq[i], None))
        else:
            break

    aligned_pairs.reverse()
    return scores_mat[first_len, second_len], aligned_pairs


align_sequences = sequences_align


@dataclass
class AccuracyStatistics:
    n_gt: int = 0
    n_asr: int = 0
    matches: int = 0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    errors: Counter = field(default_factory=Counter)

    @property
    def hits(self) -> int:
        return self.matches

    @property
    def subs(self) -> int:
        return self.substitutions

    @property
    def ins(self) -> int:
        return self.insertions

    @property
    def dels(self) -> int:
        return self.deletions

    def add_alignment(self, alignment: Iterable[Tuple[str | None, str | None]]) -> None:
        for ref_word, hyp_word in alignment:
            if ref_word is not None:
                self.n_gt += 1
            if hyp_word is not None:
                self.n_asr += 1

            if ref_word is not None and hyp_word is not None:
                if ref_word == hyp_word:
                    self.matches += 1
                else:
                    self.substitutions += 1
                    self.errors[(ref_word, hyp_word)] += 1
            elif ref_word is not None:
                self.deletions += 1
                self.errors[(ref_word, "<deleted>")] += 1
            elif hyp_word is not None:
                self.insertions += 1
                self.errors[("<inserted>", hyp_word)] += 1

    def __iadd__(self, other: "AccuracyStatistics") -> "AccuracyStatistics":
        self.n_gt += other.n_gt
        self.n_asr += other.n_asr
        self.matches += other.matches
        self.substitutions += other.substitutions
        self.insertions += other.insertions
        self.deletions += other.deletions
        self.errors.update(other.errors)
        return self

    @property
    def wer(self) -> float:
        return (self.substitutions + self.insertions + self.deletions) / self.n_gt if self.n_gt else 0.0

    @property
    def recall(self) -> float:
        return self.matches / self.n_gt if self.n_gt else 0.0

    @property
    def precision(self) -> float:
        return self.matches / self.n_asr if self.n_asr else 0.0

    @property
    def f1(self) -> float:
        if not (self.precision + self.recall):
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def f1_score(self) -> float:
        return self.f1

    def frequent_errors(self, n: int = 10):
        return self.errors.most_common(n)


def _normalize_number_token(token: str) -> str:
    if token.isdigit():
        return str(int(token))

    if token in SINGLE_NUMBER_WORDS:
        return SINGLE_NUMBER_WORDS[token]

    reduced = token
    while len(reduced) > 1 and reduced[0] in NUMBER_PREFIXES:
        candidate = reduced[1:]
        if candidate in SINGLE_NUMBER_WORDS:
            return SINGLE_NUMBER_WORDS[candidate]
        reduced = candidate

    return ORTHOGRAPHIC_VARIANTS.get(token, token)


def _collapse_numeric_tokens(tokens: list[str]) -> list[str]:
    collapsed: list[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]

        if token.isdigit() and idx + 1 < len(tokens) and tokens[idx + 1] in PERCENT_WORDS:
            collapsed.append(f"{int(token)}%")
            idx += 2
            continue

        if token in PERCENT_WORDS:
            idx += 1
            continue

        if token.isdigit():
            current = int(token)
            next_idx = idx + 1
            while next_idx < len(tokens) and tokens[next_idx].isdigit():
                nxt = int(tokens[next_idx])
                if current in (1000, 2000) and nxt < 1000:
                    current += nxt
                    next_idx += 1
                    continue
                if current >= 100 and current % 100 == 0 and nxt < 100:
                    current += nxt
                    next_idx += 1
                    continue
                if 20 <= current < 100 and current % 10 == 0 and nxt < 10:
                    current += nxt
                    next_idx += 1
                    continue
                break

            collapsed.append(str(current))
            idx = next_idx
            continue

        collapsed.append(token)
        idx += 1

    return collapsed


def normalize(text: str) -> str:
    if not text:
        return ""

    text = text.strip().lower()
    text = re.sub(r"(?<=\d)[./-](?=\d)", " ", text)
    text = re.sub(r"(\d+)\s*%", lambda match: f"{match.group(1)} \u05d0\u05d7\u05d5\u05d6", text)
    text = HEBREW_DIACRITICS_RE.sub("", text)
    text = text.translate(PUNCT_TRANSLATION)
    text = text.replace("״", "").replace("׳", "")
    for source, target in PARTNER_ADDITIONAL_REPLACEMENTS.items():
        text = text.replace(source, target)
    for source, target in PHRASE_NUMBER_WORDS.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text).strip()

    normalized_tokens: list[str] = []
    for token in text.split():
        clean_token = HEBREW_TOKEN_RE.sub("", token)
        if not clean_token:
            continue
        normalized_tokens.append(_normalize_number_token(clean_token))

    return " ".join(_collapse_numeric_tokens(normalized_tokens))


def normalize(text: str) -> str:
    if not text:
        return ""

    text = text.strip().lower()

    # Mirror the partner normalization flow exactly for this experiment.
    for char in ["-", "ג€“", "ג€”", "_", "ײ¾", "ג€™"]:
        text = text.replace(char, " ")

    text = HEBREW_DIACRITICS_RE.sub("", text)
    text = text.replace("\u05f4", "")

    for char in '.,?!:;"\'()[]{}<>ג€ג€˜ג€™ג€ג€':
        text = text.replace(char, "")

    for source, target in PARTNER_ADDITIONAL_REPLACEMENTS.items():
        text = text.replace(source, target)

    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str) -> str:
    if not text:
        return ""

    text = text.strip().lower()
    for char in ["-", "\u2013", "\u2014", "_", "\u05be", "\u2019"]:
        text = text.replace(char, " ")

    text = HEBREW_DIACRITICS_RE.sub("", text)
    text = text.replace("\u05f4", "")

    for char in '.,?!:;"\'()[]{}<>':
        text = text.replace(char, "")

    for source, target in PARTNER_ADDITIONAL_REPLACEMENTS.items():
        text = text.replace(source, target)

    return re.sub(r"\s+", " ", text).strip()


def format_alignment(alignment: Iterable[Tuple[str | None, str | None]]) -> str:
    return " | ".join(f"{ref or '<eps>'}->{hyp or '<eps>'}" for ref, hyp in alignment)


def score_texts(
    reference_text: str,
    hypothesis_text: str,
    normalize_text: bool = False,
) -> tuple[AccuracyStatistics, List[Tuple[str | None, str | None]], str, str]:
    if normalize_text:
        reference_text = normalize(reference_text)
        hypothesis_text = normalize(hypothesis_text)

    weights = WordEditWeights()
    _, alignment = sequences_align(reference_text.split(), hypothesis_text.split(), weights)
    stats = AccuracyStatistics()
    stats.add_alignment(alignment)
    return stats, alignment, reference_text, hypothesis_text


def evaluate_results_file(
    input_path: Path | str,
    output_path: Path | str,
    normalize_text: bool = False,
    alignment_log_path: Path | str | None = None,
    errors_report_path: Path | str | None = None,
) -> AccuracyStatistics:
    output_path = ensure_parent_dir(output_path)
    global_stats = AccuracyStatistics()
    alignment_log_handle = None
    alignment_writer = None

    if alignment_log_path is not None:
        alignment_log_path = ensure_parent_dir(alignment_log_path)
        alignment_log_handle = Path(alignment_log_path).open("w", encoding="utf-8", newline="")
        alignment_writer = csv.DictWriter(
            alignment_log_handle,
            fieldnames=[
                "Filename",
                "Reference Text",
                "Transcribed Text",
                "Normalized Reference",
                "Normalized Transcribed",
                "Alignment",
                "WER",
            ],
            delimiter="\t",
        )
        alignment_writer.writeheader()

    with output_path.open("w", encoding="utf-8", newline="") as target:
        fieldnames = [
            "Filename",
            "N_gt",
            "N_asr",
            "#M",
            "#S",
            "#I",
            "#D",
            "WER",
            "Recall",
            "Precision",
            "F1-Score",
        ]
        writer = csv.DictWriter(target, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for row in tsv_dict_reader(input_path):
            filename = row.get("Filename", "").strip()
            reference_text = row.get("Reference Text", "")
            hypothesis_text = row.get("Transcribed Text", "")
            file_stats, alignment, normalized_reference, normalized_hypothesis = score_texts(
                reference_text,
                hypothesis_text,
                normalize_text=normalize_text,
            )
            global_stats += file_stats

            writer.writerow(
                {
                    "Filename": filename,
                    "N_gt": file_stats.n_gt,
                    "N_asr": file_stats.n_asr,
                    "#M": file_stats.matches,
                    "#S": file_stats.substitutions,
                    "#I": file_stats.insertions,
                    "#D": file_stats.deletions,
                    "WER": f"{file_stats.wer:.4f}",
                    "Recall": f"{file_stats.recall:.4f}",
                    "Precision": f"{file_stats.precision:.4f}",
                    "F1-Score": f"{file_stats.f1:.4f}",
                }
            )

            if alignment_writer is not None:
                alignment_writer.writerow(
                    {
                        "Filename": filename,
                        "Reference Text": reference_text,
                        "Transcribed Text": hypothesis_text,
                        "Normalized Reference": normalized_reference,
                        "Normalized Transcribed": normalized_hypothesis,
                        "Alignment": format_alignment(alignment),
                        "WER": f"{file_stats.wer:.4f}",
                    }
                )

        writer.writerow(
            {
                "Filename": "TOTAL",
                "N_gt": global_stats.n_gt,
                "N_asr": global_stats.n_asr,
                "#M": global_stats.matches,
                "#S": global_stats.substitutions,
                "#I": global_stats.insertions,
                "#D": global_stats.deletions,
                "WER": f"{global_stats.wer:.4f}",
                "Recall": f"{global_stats.recall:.4f}",
                "Precision": f"{global_stats.precision:.4f}",
                "F1-Score": f"{global_stats.f1:.4f}",
            }
        )

    if alignment_log_handle is not None:
        alignment_log_handle.close()

    if errors_report_path is not None:
        errors_report_path = ensure_parent_dir(errors_report_path)
        with Path(errors_report_path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Reference", "Hypothesis", "Count"], delimiter="\t")
            writer.writeheader()
            for (ref_word, hyp_word), count in global_stats.frequent_errors(1000):
                writer.writerow({"Reference": ref_word, "Hypothesis": hyp_word, "Count": count})

    return global_stats
