## Author: David Miguel Susano Pinto <pinto@robots.ox.ac.uk>
##
## Copying and distribution of this file, with or without modification,
## are permitted in any medium without royalty provided the copyright
## notice and this notice are preserved.  This file is offered as-is,
## without any warranty.

JQ ?= jq
JUPYTEXT ?= jupytext


# First target so that `make` on its own displays the help text
help:
	@echo 'Build the Jupyter/Colab notebooks.'
	@echo ''
	@echo 'Do not edit the notebooks direcly.  Instead, change the Python'
	@echo 'scripts and run `make notebooks` to build up to date ntoebooks.'
	@echo ''
	@echo 'This requires having `jq` and `jupytext` installed locally.'
	@echo ''


## Because the notebooks are created by stdout redirection, they will
## be an empty file if the recipe fails.  Not only is this wrong, the
## empty file should not be left, after we "fix the environment" Make
## will not try to build it again because it sees the empty file as up
## to date.
.DELETE_ON_ERROR:


.PHONY: \
  help \
  notebooks


notebooks: \
  notebooks/detectors/envdante-detector.ipynb \
  notebooks/tracking/follow-things-around.ipynb

define NOTEBOOK_METADATA
{ \
  "colab": { \
    "provenance": [], \
    "toc_visible": true \
  }, \
  "kernelspec": { \
    "name": "python3", \
    "display_name": "Python 3" \
  }, \
  "language_info": { \
    "name": "python" \
  }, \
  "accelerator": "GPU", \
  "gpuClass": "standard" \
}
endef


## Jupytext adds the Jupytext options on the notebook metadata (we
## don't want them).  With regards to cell id, it places them at
## .cell[].metadata.id and generates new ones to .cell[].id.  This is
## probably because of the new nbformat 4.5.  So we manually copy the
## cell ID.
notebooks/%.ipynb: src/%.py
	$(JUPYTEXT) --to ipynb --output - $< \
	    --opt cell_metadata_filter=cellView,collapsed,-id \
	    --update-metadata '$(NOTEBOOK_METADATA)' \
	    | $(JQ) --indent 1 'del(.metadata.jupytext)' \
	    | $(JQ) --indent 1 '.cells[] |= ( .id = .metadata.id)' \
	    > $@
