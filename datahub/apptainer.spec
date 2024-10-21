Bootstrap: docker
From: ubuntu:24.04
IncludeCmd: yes
# NOTE: This apptainer was written using apptainer version `1.1.6+2-g6808b5172-ipd`
# To build this apptainer, use:
#     apptainer build --bind $PWD:/datahub_host datahub_apptainer.sif apptainer.spec

%setup
   # NOTE: This is executed on the host, not the container
   # Ensure the `GITLAB_USER` and `GITLAB_TOKEN` environment variables are set
   for var in GITLAB_USER GITLAB_TOKEN; do
      if [ -z "$(eval echo \$$var)" ]; then
         echo "ERROR: $var is not set. Please create a personal access token at https://git.ipd.uw.edu/-/user_settings/personal_access_tokens and set the GITLAB_USER and GITLAB_TOKEN environment variables."
         exit 1
      fi
   done

   # Create temporary `secrets.txt` file from host's environment variables in the container
   # (which are otherwise not available in the %post section)
   echo "Creating temporary secrets.txt file with GITLAB_USER and GITLAB_TOKEN in the container"
   touch ${APPTAINER_ROOTFS}/secrets.txt
   echo "GITLAB_USER=${GITLAB_USER}" > ${APPTAINER_ROOTFS}/secrets.txt
   echo "GITLAB_TOKEN=${GITLAB_TOKEN}" >> ${APPTAINER_ROOTFS}/secrets.txt

   # Create a directory in the container to bind the host's current working directory
   mkdir ${APPTAINER_ROOTFS}/datahub_host
   # ... for mounting `/projects` with --bind
   mkdir ${APPTAINER_ROOTFS}/projects
   # ... for mounting `/databases` with --bind
   mkdir ${APPTAINER_ROOTFS}/databases
   # ... for mounting `/squash` with --bind
   mkdir ${APPTAINER_ROOTFS}/squash

%files
   # Copy over files from the host to the container
   # Copy over the timezone info
   /etc/localtime
   # Copy over the hosts file
   /etc/hosts
   # Copy over the apt sources list
   /etc/apt/sources.list
   /archive/software/Miniconda3-latest-Linux-x86_64.sh /opt/miniconda.sh
   environment.yaml /opt/environment.yaml

%post
   # Perform the bulk of the build steps within the container

   # Check if /projects is accessible
   if [ ! -d "/projects" ]; then
      echo "ERROR: /projects directory is not accessible, but needed for tests."
      echo "Make sure you bind the /projects directory when you run the container."
      echo "The easiest way to ensure this is to just use \`make apptainer\`,"
      echo "which has the correct flags set up for you."
      exit 1
   fi
   echo "Verified that /projects directory is accessible."

   # Check if /databases is accessible
   if [ ! -f "/databases/TrRosetta/cif/oc/5ocm.cif.gz" ]; then
      echo "ERROR: /databases directory is not accessible, but needed for tests."
      echo "Make sure you bind the /databases directory when you run the container."
      echo "The easiest way to ensure this is to just use \`make apptainer\`,"
      echo "which has the correct flags set up for you."
      exit 1
   fi
   echo "Verified that /databases directory is accessible."

   # Check if /squash is accessible
   if [ ! -d "/squash/af2_distillation_facebook" ]; then
      echo "ERROR: /squash directory is not accessible, but needed for tests."
      echo "Make sure you bind the /squash directory when you run the container."
      exit 1
   fi
   echo "Verified that /squash directory is accessible."

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
   # ... temporarily set the `GITLAB_USER` and `GITLAB_TOKEN` environment variables
   #     from the secrets file
   echo "Exporting GITLAB_USER and GITLAB_TOKEN from secrets.txt"
   export GITLAB_USER=$(grep GITLAB_USER /secrets.txt | cut -d '=' -f2)
   export GITLAB_TOKEN=$(grep GITLAB_TOKEN /secrets.txt | cut -d '=' -f2)
   # ... remove secrets file
   rm secrets.txt
   # ... verify that the secrets file is not present on the container
   if [ -e /secrets.txt ]; then
      echo "ERROR: secrets.txt is still present on the container"
      exit 1
   else
      echo "Verified that secrets.txt is not present on the container"
   fi
   # ... verify that the `GITLAB_USER` and `GITLAB_TOKEN` environment variables are set
   for var in GITLAB_USER GITLAB_TOKEN; do
      if [ -z "$(eval echo \$$var)" ]; then
         echo "ERROR: $var is not set"
         exit 1
      fi
   done
   echo "Verified that GITLAB_USER and GITLAB_TOKEN are set"

   ## GENERAL SETUP
   # Switch shell to bash
   ln -sf /bin/bash /bin/sh

   # Common symlinks (within container)
   ln -s /net/software /software
   ln -s /home /mnt/home
   ln -s /projects /mnt/projects
   ln -s /net /mnt/net

   ## PACKAGE INSTALLATION
   apt-get update
   # Install build essentials and other required packages (needed for compiling biotite cython files)
   apt-get install -y build-essential gcc g++
   # Install make (so we can run `make format`, `make clean`, etc.)
   apt-get install -y make
   # required X libs
   apt-get install -y libx11-6 libxau6 libxext6 libxrender1
   #  git
   apt-get install -y git
   apt-get install -y libaio-dev
   apt-get clean

   ## ENVIRONMENT CREATION & DEPENDENCY INSTALLATION
   # Install conda
   bash /opt/miniconda.sh -b -u -p /usr

   # install everything
   # ... the environment in the container is at `/usr/envs/datahub-apptainer`
   conda env create --file /opt/environment.yaml --name datahub-apptainer

   # Set up conda
   conda init bash
   # Add conda environment to PATH
   export PATH=/usr/envs/datahub-apptainer/bin:$PATH

   ## CLEANUP
   # Unset the `GITLAB_USER` and `GITLAB_TOKEN` environment variables to avoid possibly
   # leaking them in the container
   unset GITLAB_USER
   unset GITLAB_TOKEN
   # ... verify that the `GITLAB_USER` and `GITLAB_TOKEN` environment variables are unset
   for var in GITLAB_USER GITLAB_TOKEN; do
      if [ -n "$(eval echo \$$var)" ]; then
         echo "ERROR: $var is still set"
         exit 1
      fi
   done
   echo "Verified that GITLAB_USER and GITLAB_TOKEN are unset."

   # clean up files to reduce size
   # ... remove conda
   conda clean -a -y
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
   #  Run this manually with: apptainer test datahub_apptainer.sif
   echo "Container OS: $(grep '^PRETTY_NAME=' /etc/os-release | cut -d '=' -f2- | sed 's/"//g')"
   echo "... running the $(readlink /proc/$$/exe) shell."
   echo "... conda at $(which conda)"

   # Verify that `secrets.txt` is not present on the container
   if [ -e /secrets.txt ]; then
      echo "ERROR: secrets.txt is still present on the container"
      exit 1
   fi
   echo "Verified that secrets.txt is not present in the container"
   
   # activate environment (the .bashrc in the container is at `/root/.bashrc`)
   source /usr/etc/profile.d/conda.sh
   conda activate datahub-apptainer
   echo "... running python $(python --version) at $(which python)"

   # ... check that python is run from the datahub-apptainer environment
   DESIRED_PATH="/usr/envs/datahub-apptainer/bin/python"
   if [ "$(which python)" != "$DESIRED_PATH" ]; then
      echo "ERROR: python is not running from the datahub-apptainer environment"
      exit 1
   fi

   # ... check that pytest runs for datahub tests
   echo "Basic build succeeded. For a proper test, run: apptainer run --bind $PWD:/datahub_host datahub_apptainer.sif"

   # Run make test
   # ... add the `datahub` path to the PYTHONPATH environment variable
   export PYTHONPATH=/datahub_host:$PYTHONPATH
   echo "Running make test"
   cd /datahub_host && make test

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

   To build this apptainer, use:
      apptainer build --bind $PWD:/datahub_host path/to/apptainer.sif apptainer.spec

   To run the container, use:
      apptainer exec /path/to/apptainer.sif <command>
      OR
      ./path/to/apptainer.sif <command>

   To get an interactive shell in the container, use:
      apptainer shell /path/to/apptainer.sif

%labels
    Author simon.mathis@gmail.com, ncorley@uw.edu
    Version v0.0.2
    ApptainerVersion 1.1.6+2-g6808b5172-ipd
