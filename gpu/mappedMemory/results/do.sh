gnuplot plot-time.pg
ps2pdf time.ps
pdfcrop time.pdf --margin 10
mv time-crop.pdf time.pdf
if [ $# -eq 0 ]
  then
	open time.pdf &
fi
