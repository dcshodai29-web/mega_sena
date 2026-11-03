.PHONY: install test data check update analyze backtest predict score verify report all

install:
	pip install -e .[dev]

test:
	pytest -q

data:            ## baixa/usa cache das fontes, confere SHA-256 e reconstrói a base
	megalab data build

check:           ## reconstrói a base a partir dos extratos versionados
	megalab data check

update:          ## busca concursos novos (API da Caixa + espelhos)
	megalab data update

analyze:
	megalab analyze

backtest:
	megalab backtest

predict:
	megalab ledger predict

score:
	megalab ledger score

verify:
	megalab ledger verify

report:
	megalab report

all: check analyze backtest score report
