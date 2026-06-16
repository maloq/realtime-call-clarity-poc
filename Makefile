.PHONY: test lint format smoke

test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests

smoke:
	callclarity eval data.input_dir=tests/fixtures output_dir=outputs/smoke pipeline=denoise_agc data.max_files=1
