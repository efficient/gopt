gnuplot plot-microbenchmarks.pg
ps2pdf microbenchmarks.ps
pdfcrop microbenchmarks.pdf
mv microbenchmarks-crop.pdf microbenchmarks.pdf
if [ $# -eq 0 ]
  then
	open microbenchmarks.pdf &
fi
