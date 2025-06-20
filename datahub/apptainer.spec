Bootstrap: docker
From: ubuntu:24.04
IncludeCmd: yes
# NOTE: This apptainer was written using apptainer version `1.1.6+2-g6808b5172-ipd`
# To build this apptainer, use:
#     apptainer build --bind $PWD:/datahub_host datahub_apptainer.sif apptainer.spec

%setup
   # NOTE: This is executed on the host, not the container
   # Ensure the token environment variables are set
   set +x  # ... supress bash output to avoid printing the tokens in the output
   for var in GITHUB_USER GITHUB_TOKEN; do
      if [ -z "$(eval echo \$$var)" ]; then
         set -x
         echo "ERROR: $var is not set. Please create a personal access token at" 
         echo "  - GitHub: https://github.com/settings/tokens"
         echo "Then set the following environment variables:"
         echo "  - GITHUB_USER"
         echo "  - GITHUB_TOKEN"
         exit 1
      fi
   done
   set -x
   # Create temporary `secrets.txt` file from host's environment variables in the container
   # (which are otherwise not available in the %post section)
   echo "Creating temporary secrets.txt file with access tokens in the container"
   set +x
   touch ${APPTAINER_ROOTFS}/secrets.txt
   echo "GITHUB_USER=${GITHUB_USER}" >> ${APPTAINER_ROOTFS}/secrets.txt
   echo "GITHUB_TOKEN=${GITHUB_TOKEN}" >> ${APPTAINER_ROOTFS}/secrets.txt
   set -x

%files
   environment.yaml /opt/environment.yaml

%post
   # get os name
   echo "Running on OS name $(lsb_release -i | awk '{ print $3 }')"
   # get os version
   echo "... in OS version $(lsb_release -r | awk '{ print $2 }')"

   ## SECRETS FILE
   # Deal with secrets file
   # ... verify that the secrets file is present on the container
   if [ ! -e /secrets.txt ]; then
      echo "ERROR: secrets.txt is not present on the container"
      exit 1
   fi
   # ... temporarily set the access token environment variables
   #     from the secrets file
   echo "Exporting access tokens from secrets.txt"
   set +x
   export GITHUB_USER=$(grep GITHUB_USER /secrets.txt | cut -d '=' -f2)
   export GITHUB_TOKEN=$(grep GITHUB_TOKEN /secrets.txt | cut -d '=' -f2)
   set -x
   # ... remove secrets file
   rm secrets.txt
   # ... verify that the secrets file is not present on the container
   if [ -e /secrets.txt ]; then
      echo "ERROR: secrets.txt is still present on the container"
      exit 1
   else
      echo "Verified that secrets.txt is not present on the container"
   fi
   # ... verify that the access token environment variables are set
   set +x
   for var in GITHUB_USER GITHUB_TOKEN; do
      if [ -z "$(eval echo \$$var)" ]; then
         echo "ERROR: $var is not set"
         exit 1
      fi
   done
   set -x
   echo "Verified that access tokens are set"

   ## GENERAL SETUP
   # Switch shell to bash
   ln -sf /bin/bash /bin/sh

   # Common symlinks (within container)
   ln -s /net/databases /databases
   ln -s /net/software /software
   ln -s /home /mnt/home
   ln -s /projects /mnt/projects
   ln -s /net /mnt/net

   ## PACKAGE INSTALLATION
   apt-get update
   # Install build essentials and other required packages (needed for compiling biotite cython files)
   apt-get install -y build-essential gcc g++
   # Install make (so we can run `make format`, `make clean`, etc.)
   apt-get install -y make git wget libaio-dev
   # required X libs
   apt-get install -y libx11-6 libxau6 libxext6 libxrender1
   apt-get clean

   ## ENVIRONMENT CREATION & DEPENDENCY INSTALLATION
   # Download miniconda
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda.sh

   # Install conda
   bash /opt/miniconda.sh -b -u -p /usr

   # install everything
   # ... the environment in the container is at `/usr/envs/datahub-apptainer`
   conda install -c conda-forge mamba
   mamba env create --file /opt/environment.yaml --name datahub-apptainer

   # Set up conda
   conda init bash
   # Add conda environment to PATH
   export PATH=/usr/envs/datahub-apptainer/bin:$PATH

   ## CLEANUP
   # Unset the access token environment variables to avoid possibly
   # leaking them in the container
   unset GITHUB_USER
   unset GITHUB_TOKEN
   # ... verify that the access token environment variables are unset
   set +x
   for var in GITHUB_USER GITHUB_TOKEN; do
      if [ -n "$(eval echo \$$var)" ]; then
         set -x
         echo "ERROR: $var is still set"
         exit 1
      fi
   done
   set -x
   echo "Verified that access tokens are unset."

   # clean up files to reduce size
   # ... remove conda
   mamba clean -a -y
   # ... remove other apt packages that are no longer needed
   apt-get -y purge build-essential git wget
   apt-get -y autoremove
   apt-get clean
   rm /opt/miniconda.sh

%test
   #!/bin/bash
   # NOTE: The %test section is automatically run after the apptainer build process.
   #  It is intended to be executed separately using the apptainer test command.
   #  You can use this to verify the integrity of the container.
   #  Run this manually with: apptainer test cifutils_apptainer.sif
   echo "Container OS: $(grep '^PRETTY_NAME=' /etc/os-release | cut -d '=' -f2- | sed 's/"//g')"
   echo "... running the $(readlink /proc/$$/exe) shell."
   echo "... conda at $(command -v conda)"

   # activate environment (the .bashrc in the container is at `/root/.bashrc`)
   source /usr/etc/profile.d/conda.sh
   conda activate datahub-apptainer
   echo "... running python $(python --version) at $(command -v python)"

   # ... check that python is run from the cifutils-apptainer environment
   DESIRED_PATH="/usr/envs/datahub-apptainer/bin/python"
   if [ "$(command -v python)" != "$DESIRED_PATH" ]; then
      echo "ERROR: python is not running from the datahub-apptainer environment"
      exit 1
   fi

   # Run make test
   echo "Running make test"
   make test   

   echo "All tests completed."

%environment
   source /usr/etc/profile.d/conda.sh
   conda activate datahub-apptainer

%runscript
   # NOTE: The %runscript is invoked when the container is run without specifying a different command. 
   exec "$@"

%help
   datahub environment for running datahub independently and for development

   To see this help message, use:
      apptainer run-help datahub_apptainer.sif

   To run the container, use:
      apptainer exec /path/to/apptainer.sif <command>
      OR
      ./path/to/apptainer.sif <command>

   To get an interactive shell in the container, use:
      apptainer shell /path/to/apptainer.sif

%labels
    Author simon.mathis@gmail.com, ncorley@uw.edu
    Version v1.0.0
    ApptainerVersion 1.1.6+2-g6808b5172-ipd
