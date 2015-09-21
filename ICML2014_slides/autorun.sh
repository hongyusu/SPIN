


name='slides'

pdflatex $name
bibtex $name
bibtex $name
pdflatex $name

rm $name.toc
rm $name.aux
rm $name.blg
rm $name.log
rm $name.snm
rm $name.nav
rm $name.out

mv $name.pdf ICML2014_slides.pdf
