TF      := terraform -chdir=terraform
BUILD   := terraform/.build/api_src

.PHONY: build plan apply output smoke clean test

# Lambda bundle: handler + Linux x86_64 wheels for `cryptography`, fetched as
# prebuilt binaries so this works from macOS without Docker.
build:
	rm -rf $(BUILD) && mkdir -p $(BUILD)
	cp lambda/api/handler.py $(BUILD)/
	python3 -m pip install --quiet --target $(BUILD) \
		--platform manylinux2014_x86_64 --implementation cp --python-version 3.12 \
		--only-binary=:all: -r lambda/api/requirements.txt

plan: build
	$(TF) init -input=false
	$(TF) plan -out tfplan

apply:
	$(TF) apply tfplan

output:
	$(TF) output

# End-to-end test of every model on the deployed GPU instance (~30-60 min).
smoke:
	python3 cli/vidgen.py smoke

clean:
	rm -rf terraform/.build terraform/tfplan

# Unit tests (no cloud, no GPU): each file runs in its own process.
test:
	@for f in tests/test_*.py; do python3 -B $$f || exit 1; done
