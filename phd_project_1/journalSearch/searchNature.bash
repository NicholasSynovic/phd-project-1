#!/bin/bash

currentYear=$(date +%Y)

for year in $(seq 2015 $currentYear); do
    python3.10 nature.py --search-query '"HuggingFace"' --start-year $year --end-year $year
    python3.10 nature.py --search-query '"Hugging Face"' --start-year $year --end-year $year
    python3.10 nature.py --search-query '"Deep Neural Network"' --start-year $year --end-year $year
done
