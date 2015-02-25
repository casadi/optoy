test:
	py.test --cov optoy --doctest-modules optoy tests
	python ipnbdoctest.py examples/*.ipynb
