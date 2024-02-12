#!/bin/bash

source optparse/optparse.bash
optparse.define short=i long=input desc="Directory to recursively search for PDF (.pdf) fils" variable=pdfDirectory
source $( optparse.build )

if [ "$pdfDirectory" == "" ]; then
    echo "ERROR: Please provide an input"
    exit 1
fi

# readarray -t pdfs < <(find $pdfDirectory -type f -name "*.pdf")
# pdfsCount=${#pdfs[@]}

# for ((i=0; i<$pdfsCount; i++)); do
#     currentPDF="${pdfs[$i]}"
#     python ../phd_project_1/pdf2txt.py -i $(printf %q $currentPDF) -o ../data/textFiles/pdfText_$currentPDF.txt
# done

find $pdfDirectory -type f -name "*.pdf" | parallel -I @ --bar python ../phd_project_1/pdf2txt.py -i @ -o ../data/textFiles/pdfText_{#}.txt
