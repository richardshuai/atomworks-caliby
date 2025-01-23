Bootstrap: docker
From: ubuntu:24.04
IncludeCmd: yes
# NOTE: This apptainer was written using apptainer version `1.1.6+2-g6808b5172-ipd`
# To build this apptainer, use:
#     apptainer build --bind $PWD:/cifutils_host cifutils_apptainer.sif apptainer.spec

%setup
   # Create a directory in the container to bind the host's current working directory
   mkdir ${APPTAINER_ROOTFS}/cifutils_host

%files
   environment.yaml /opt/environment.yaml

%post
   # get os name
   echo "Running on OS name $(lsb_release -i | awk '{ print $3 }')"
   # get os version
   echo "... in OS version $(lsb_release -r | awk '{ print $2 }')"

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
   # Install make (so we can run `make format`, `make clean`, etc.), git, and wget
   apt-get install -y make git wget
   # required X libs
   apt-get install -y libx11-6 libxau6 libxext6 libxrender1
   apt-get install -y libaio-dev
   apt-get clean

   ## ENVIRONMENT CREATION & DEPENDENCY INSTALLATION
   # Download and install Miniconda
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda.sh
   chmod +x /opt/miniconda.sh

   # Install conda
   bash /opt/miniconda.sh -b -u -p /usr

   # install everything
   # ... the environment in the container is at `/usr/envs/cifutils-apptainer`
   conda env create --file /opt/environment.yaml --name cifutils-apptainer

   # Set up conda
   conda init bash
   # Add conda environment to PATH
   export PATH=/usr/envs/cifutils-apptainer/bin:$PATH

   # Run the biotite setup command
   # (Temporary measure until we switch to released Biotite version)
   . /usr/etc/profile.d/conda.sh
   conda activate cifutils-apptainer
   python -m biotite.setup_ccd

   ## CLEANUP
   # clean up files to reduce size
   # ... remove conda
   conda clean -a -y
   # ... remove other apt packages that are no longer needed
   apt-get -y purge build-essential wget
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
   conda activate cifutils-apptainer
   echo "... running python $(python --version) at $(command -v python)"

   # ... check that python is run from the cifutils-apptainer environment
   DESIRED_PATH="/usr/envs/cifutils-apptainer/bin/python"
   if [ "$(command -v python)" != "$DESIRED_PATH" ]; then
      echo "ERROR: python is not running from the cifutils-apptainer environment"
      exit 1
   fi

   # Run make test
   echo "Running make test"
   make test   

   echo "All tests completed."

%environment
   source /usr/etc/profile.d/conda.sh
   conda activate cifutils-apptainer

%runscript
   # NOTE: The %runscript is invoked when the container is run without specifying a different command. 
   exec "$@"

%help
   cifutils environment for running cifutils independently and for development

   To see this help message, use:
      apptainer run-help cifutils_apptainer.sif

   To run the container, use:
      apptainer exec /path/to/cifutils_apptainer.sif <command>
      OR
      ./path/to/cifutils_apptainer.sif <command>

   To get an interactive shell in the container, use:
      apptainer shell /path/to/cifutils_apptainer.sif

%labels
    Authors simon.mathis@gmail.com, ncorley@uw.edu
    Version v2.0.0
    ApptainerVersion 1.1.6+2-g6808b5172-ipd
