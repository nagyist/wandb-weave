generate_service_api_docs:
	mkdir -p ./docs/reference/service-api
	rm -rf ./docs/reference/service-api
	mkdir -p ./docs/reference/service-api
	python scripts/generate_service_api_spec.py
	yarn docusaurus gen-api-docs all

generate_python_sdk_docs:
	mkdir -p ./docs/reference/python-sdk
	rm -rf ./docs/reference/python-sdk
	mkdir -p ./docs/reference/python-sdk
	python scripts/generate_python_sdk_docs.py

generate_typescript_sdk_docs:
	mkdir -p ./docs/reference/typescript-sdk
	rm -rf ./docs/reference/typescript-sdk
	mkdir -p ./docs/reference/typescript-sdk
	bash scripts/generate_typescript_sdk_docs.sh

generate_notebooks_docs:
	mkdir -p ./docs/reference/gen_notebooks
	rm -rf ./docs/reference/gen_notebooks
	mkdir -p ./docs/reference/gen_notebooks
	python scripts/generate_notebooks.py

generate_doc_images:
	python scripts/generate_images.py

# Per discussion with Tim, generate_doc_images is a little hacky and fragile still.
# It will need to be run manually until it's stable enough to include in generate_all
generate_all: generate_service_api_docs generate_python_sdk_docs generate_notebooks_docs
	yarn build

update_playground_models:
	python scripts/update_playground_models.py

generate_llms_docs:
	mkdir -p ./docs/static
	python docs/scripts/generate_llmstxt.py
	python docs/scripts/generate_llmstxt.py --categories
