gnuplot plot-chained-memaccess.pg
ps2pdf chained-memaccess.ps
pdfcrop --margins 10 chained-memaccess.pdf 
mv chained-memaccess-crop.pdf chained-memaccess.pdf
if [ $# -eq 0 ]
  then
	open chained-memaccess.pdf &
fi
