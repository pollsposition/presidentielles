.PHONY: build clean

build: _build/premier-tour.html

_build/premier-tour.html:
	mkdir -p $(dir $@)
	python src/premier_tour.py

clean:
	rm -r _build
