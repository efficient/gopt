gnuplot plot-randmem-comp.pg
ps2pdf randmem-comp.ps
pdfcrop --margin 10 randmem-comp.pdf
mv randmem-comp-crop.pdf randmem-comp.pdf
if [ $# -eq 0 ]
  then
	open randmem-comp.pdf &
fi
