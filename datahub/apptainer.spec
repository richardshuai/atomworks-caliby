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
   rm /bin/sh; ln -s /bin/bash /bin/sh

   # Common symlinks
   ln -s /net/databases /databases
   ln -s /net/software /software
   ln -s /home /mnt/home
   ln -s /projects /mnt/projects
   ln -s /net /mnt/net

   ## PACKAGE INSTALLATION
   apt-get update
   # Install build essentials and other required packages (needed for compiling biotite cython files)
   apt-get install -y build-essential gcc g++
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

   # Set up conda and activate the datahub-apptainer environment
   conda init bash
   # Add conda environment to PATH
   echo "source /usr/etc/profile.d/conda.sh" >> /root/.bashrc
   echo "source /usr/bin/activate" >> /root/.bashrc
   echo "conda activate datahub-apptainer" >> /root/.bashrc
   echo "export PATH=/usr/envs/datahub-apptainer/bin:$PATH" >> /root/.bashrc
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
   #  Run this manually with: apptainer test cifutils_apptainer.sif
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
   source /usr/bin/activate
   conda activate datahub-apptainer
   echo "... running python $(python --version) at $(which python)"

   # ... check that python is run from the datahub-apptainer environment
   DESIRED_PATH="/usr/envs/datahub-apptainer/bin/python"
   if [ "$(which python)" != "$DESIRED_PATH" ]; then
      echo "ERROR: python is not running from the datahub-apptainer environment"
      exit 1
   fi

   # ... check that pytest runs for datahub tests
   echo "Basic build succeeded. For a proper test, run: apptainer run --bind $PWD:/cifutils_host cifutils_apptainer.sif"

%environment
   source /usr/etc/profile.d/conda.sh
   source /usr/bin/activate
   conda activate datahub-apptainer
   source /root/.bashrc

%runscript
   # NOTE: The %runscript is invoked when the container is run without specifying a different command. 
   exec "$@"

%help
   datahub environment for running datahub independently and for development

   To see this help message, use:
      apptainer run-help cifutils_apptainer.sif

   To build this apptainer, use:
      apptainer build --bind $PWD:/cifutils_host path/to/cifutils_apptainer.sif apptainer.spec

   To run the container, use:
      apptainer exec /path/to/cifutils_apptainer.sif <command>
      OR
      ./path/to/cifutils_apptainer.sif <command>

   To get an interactive shell in the container, use:
      apptainer shell /path/to/cifutils_apptainer.sif

%labels
    Author smathis@uw.edu
    Version v0.0.1
    ApptainerVersion 1.1.6+2-g6808b5172-ipd
