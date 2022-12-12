#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linopy module for defining constant values used within the package.
"""

EQUAL = "="
GREATER_EQUAL = ">="
LESS_EQUAL = "<="

long_EQUAL = "=="
short_GREATER_EQUAL = ">"
short_LESS_EQUAL = "<"


SIGNS = {EQUAL, GREATER_EQUAL, LESS_EQUAL}
SIGNS_alternative = {long_EQUAL, short_GREATER_EQUAL, short_LESS_EQUAL}

sign_replace_dict = {
    long_EQUAL: EQUAL,
    short_GREATER_EQUAL: GREATER_EQUAL,
    short_LESS_EQUAL: LESS_EQUAL,
}
