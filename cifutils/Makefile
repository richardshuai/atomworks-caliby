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
	ruff check --fix ./cifutils/cifutils_biotite
	ruff check --fix .

_biotite_access_error:
	@echo "==============================================================================="; \
	echo "Error: Environment variables BIOTITE_INSTALL_TOKEN or GITLAB_USER and GITLAB_TOKEN must be set."; \
	echo ""; \
	echo "Option 1: Set the environment variable BIOTITE_INSTALL_TOKEN. You can ask one of the maintainers for this token."; \
	echo "Option 2: Set the environment variables GITLAB_USER and GITLAB_TOKEN. You can create a personal access token on GitLab."; \
	echo ""; \
	echo "To expose these variables, you can use:"; \
	echo "export BIOTITE_INSTALL_TOKEN=<gitlab-token>"; \
	echo ""; \
	echo " ... or ... "; \
	echo ""; \
	echo "export GITLAB_USER=<gitlab-username>"; \
	echo "export GITLAB_TOKEN=<gitlab-token>"; \
	echo ""; \
	echo "It is recommended that you set these tokens in your .bashrc or .zshrc file for future use."; \
	echo "===============================================================================";
	exit 1;


## Install the production version of the code
install:
	@echo "Installing production version"
	@pip install --upgrade git+https://biotite-install:${BIOTITE_INSTALL_TOKEN}@git.ipd.uw.edu/ai/biotite.git || \
		pip install --upgrade git+https://${GITLAB_USER}:${GITLAB_TOKEN}@git.ipd.uw.edu/ai/biotite.git || \
		$(MAKE) _biotite_access_error
	@pip install .



## Install the development version of the code
install-dev:
	@echo "Installing development version"
	@${CONDA_BINARY} install -c conda-forge openbabel || { \
		echo "Failed to install openbabel with conda, attempting fallback installation..."; \
		echo "... ignoring for now as it is only needed for CCD data pre-computation ..."; \
		exit 1; \
	}
	@pip install --upgrade git+https://biotite-install:${BIOTITE_INSTALL_TOKEN}@git.ipd.uw.edu/ai/biotite.git || \
		pip install --upgrade git+https://${GITLAB_USER}:${GITLAB_TOKEN}@git.ipd.uw.edu/ai/biotite.git || \
		$(MAKE) _biotite_access_error
	@pip install -e ".[dev]"

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