.PHONY: init clean format

#################################################################################
# GLOBALS                                                                       #
#################################################################################

# Set the project directory
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# Set conda/mamba binary, mamba if available, conda otherwise
CONDA_BINARY := $(shell command -v mamba || command -v conda)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Format src directory using black
format:
	ruff format .
	ruff check --fix .

_token_access_error:
	@echo "==============================================================================="; \
	echo "Error: Environment variables GITLAB_USER and GITLAB_TOKEN must be set."; \
	echo ""; \
	echo "You need to set the environment variables GITLAB_USER and GITLAB_TOKEN."; \
	echo "You can create a personal access token on GitLab at:"; \
	echo "   https://git.ipd.uw.edu/-/user_settings/personal_access_tokens"; \
	echo ""; \
	echo "To expose these variables, you can use:"; \
	echo "export GITLAB_USER=<gitlab-username>"; \
	echo "export GITLAB_TOKEN=<gitlab-token>"; \
	echo ""; \
	echo "It is recommended that you set these tokens in your .bashrc or .zshrc file for future use."; \
	echo "===============================================================================";
	exit 1;

## Create a new conda environment and install cifutils
env:
	@echo "Creating cifutils conda environment: `cifutils`"
	# Check if the GITLAB_USER and GITLAB_TOKEN are set
	if [ -z "$(GITLAB_USER)" ] || [ -z "$(GITLAB_TOKEN)" ]; then \
		$(MAKE) _token_access_error; \
	fi
	$(CONDA_BINARY) env create -n cifutils --file environment.yaml
	conda init
	conda activate cifutils
	pip install -e ".[dev]"

## Install cifutils locally into the current environment
install:
	@echo "Installing cifutils"
	# Check if the GITLAB_USER and GITLAB_TOKEN are set
	if [ -z "$(GITLAB_USER)" ] || [ -z "$(GITLAB_TOKEN)" ]; then \
		$(MAKE) _token_access_error; \
	fi
	# Install the conda requirements in the current activated environment
	$(CONDA_BINARY) env update --file environment.yaml
	# Install the pip requirements in the current activated environment
	pip install -e ".[dev]"

## Build the apptainer image
apptainer:
	# Check if the GITLAB_USER and GITLAB_TOKEN are set
	if [ -z "$(GITLAB_USER)" ] || [ -z "$(GITLAB_TOKEN)" ]; then \
		$(MAKE) _token_access_error; \
	fi
	apptainer build --bind $(PWD):/cifutils_host cifutils.sif apptainer.spec

## Run pytest and generate coverage report
test:
	pytest --cov=cifutils --cov-report=term-missing --cov-report=html --cov-report=xml tests/

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')