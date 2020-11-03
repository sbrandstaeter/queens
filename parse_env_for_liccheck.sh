#!/bin/bash
grep "^ *-" environment.yml > requirements.tmp
sed -i  's/^ *-//' requirements.tmp  
sed -i  '/defaults/d' requirements.tmp 
sed -i  '/conda-forge/d' requirements.tmp
sed -i  '/python/d' requirements.tmp
sed -i  '/cython/d' requirements.tmp
sed -i  '/pip/d' requirements.tmp
sed -i  '/pytest/d' requirements.tmp
