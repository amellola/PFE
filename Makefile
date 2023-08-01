.PHONY: help venv install update_req clean clean_full
default: help

-include makefiles/Makefile.*
## shows this help
help:
		@printf "Available targets:\n\n"
		@awk '/^[a-zA-Z\-\_0-9%:\\]+/ { \
		  helpMessage = match(lastLine, /^## (.*)/); \
		  if (helpMessage) { \
			helpCommand = $$1; \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
      gsub("\\\\", "", helpCommand); \
      gsub(":+$$", "", helpCommand); \
			printf "  \x1b[32;01m%-35s\x1b[0m %s\n", helpCommand, helpMessage; \
		  } \
		} \
		{ lastLine = $$0 }' $(MAKEFILE_LIST) | sort -u
		@printf "\n"
## enter virtual environment
venv:
	poetry shell
## install dependencies
install: pyproject.toml
	poetry install
## update requirements.txt file
update_req: pyproject.toml poetry.lock
	poetry export -f requirements.txt --output requirements.txt --without-hashes
## watch nvidia gpu usage
nvidia:
	watch -n0.1 nvidia-smi
## clean environment (remove cache)
clean:
	py3clean .
	rm -rf .mypy_cache
	rm -rf .pytest_cache
## clean environment (remove venv and polyaxon-config as well)
clean_full:
	py3clean .
	rm -rf .mypy_cache
	rm -rf .polyaxon
	rm -rf .pytest_cache
	rm -rf .venv
