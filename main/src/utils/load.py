#!/usr/bin/env python
# -*- coding:utf-8 -*-

# dependency
# public
import json

def load_json(path: str) -> list:

	with open(path, 'r', encoding='utf-8') as f:
		return json.load(f)

def load_txt(path: str) -> list:

      with open(path, 'r', encoding='utf-8') as f:
            return f.read().splitlines()