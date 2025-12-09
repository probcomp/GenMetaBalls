#!/bin/bash

pdflatex plate_diagram.tex
convert -verbose -density 300 -trim plate_diagram.pdf -size 1080x1080 -quality 100 -flatten -sharpen 0x1.0 plate_diagram.png
