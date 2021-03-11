SRC=test.tex
OPTIONS=-pdf -bibtex
TC=latexmk

all:
	$(TC) $(OPTIONS) $(SRC)

clean:
	latexmk -c

purge:
	latexmk -C
