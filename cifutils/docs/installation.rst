Installation
============

cifutils can be installed in several ways, depending on your workflow and environment. Below are the recommended methods:

1. Using the Standalone Apptainer (Recommended)
-----------------------------------------------
This is ideal for dataset parsing and generation in a controlled environment.

.. code-block:: bash

   # Set up IPD-specific environment variables
   source ./.ipd/setup.sh
   # Use the provided apptainer image
   ./.ipd/cifutils.sif

2. Local Conda Environment
--------------------------
For development and testing:

.. code-block:: bash

   git clone git@git.ipd.uw.edu:ai/cifutils.git
   cd cifutils
   make install  # or pip install -e "."

   # Create a .env file (see .env.sample) with CCD and PDB paths as needed

To install in a fresh environment:

.. code-block:: bash

   git clone git@git.ipd.uw.edu:ai/cifutils.git
   cd cifutils
   # Set up Gitlab credentials in your shell
   echo 'export GITLAB_USER=<Gitlab_Username>' >> .bashrc
   echo 'export GITLAB_TOKEN=<Gitlab_PAT_Token>' >> .bashrc
   source .bashrc
   make env
   pytest tests

3. As a Dependency in Your Apptainer
------------------------------------
Add `cifutils/src` to your apptainer's PYTHONPATH:

.. code-block:: bash

   export PYTHONPATH=$PWD/src:$PYTHONPATH

Or, if at IPD:

.. code-block:: bash

   source ./.ipd/setup.sh

For new apptainers, see the apptainer.spec file for integration details. 