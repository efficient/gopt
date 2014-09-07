TARGET=paper
all: pdf

pdf:
	GS_OPTIONS=-dPDFSETTINGS=/prepress rubber --pdf -Wrefs -Wmisc paper
	$(info If figures don't appear, use make fig)$

fig:
	./do.sh fig

fig-clean:
	./do.sh fig-clean

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *~ *.bak $(TARGET).ps $(TARGET).pdf $(TARGET).dvi

ps: pdf
	GS_OPTIONS=-dPDFSETTINGS=/prepress pdftops -level1 $(TARGET).pdf
