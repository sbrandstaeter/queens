#!/bin/bash
grep "^ *-" environment.yml > requirements.tmp
sed -i '.bak' 's/^ *-//' requirements.tmp  
sed -i '.bak' '/defaults/d' requirements.tmp 
sed -i '.bak' '/conda-forge/d' requirements.tmp
sed -i '.bak' '/python/d' requirements.tmp
sed -i '.bak' '/cython/d' requirements.tmp
sed -i '.bak' '/pip/d' requirements.tmp
