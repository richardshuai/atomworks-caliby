##################################################################################
### HELPER FUNCTIONS
##################################################################################

check_is_correct_root_dir() {
    local root_dir="$1"
    local required_dirs=("src")
    local required_files=("src/datahub/__init__.py")
    local missing_dirs=()
    local missing_files=()

    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$root_dir/$dir" ]; then
            missing_dirs+=("$dir")
        fi
    done

    for file in "${required_files[@]}"; do
        if [ ! -f "$root_dir/$file" ]; then
            missing_files+=("$file")
        fi
    done

    if [ ${#missing_dirs[@]} -ne 0 ] || [ ${#missing_files[@]} -ne 0 ]; then
        # Uncomment this for debugging: 
        # echo "ERROR! When checking if $root_dir is the correct root directory, the following directories were missing:"
        # printf '  - %s\n' "${missing_dirs[@]}"
        return 1
    fi
    return 0
}


check_and_set_env_var() {
    local var_name="$1"
    local var_value="$2"
    local env_file=".env"
    local to_set=true

    if grep -q "^${var_name}=" "$env_file"; then
        # Extract the value of the variable from the .env file
        local existing_value=$(grep "^${var_name}=" "$env_file" | cut -d '=' -f2-)
        if [ -z "$existing_value" ]; then
            echo "WARNING! ${var_name} is set in the .env file but is empty. Removing it from the .env file."
            sed -i "/^${var_name}=/d" "$env_file"
            to_set=true
        else
            echo "... ${var_name} already set in your .env file and is not empty"
            to_set=false
        fi
    fi

    if $to_set; then
        echo "... setting ${var_name} in your .env file"
        echo "" >> "$env_file"
        echo "${var_name}=${var_value}" >> "$env_file"
    fi
}

##################################################################################
### MAIN SCRIPT
##################################################################################

# Initialize FOUND_ROOT_DIR as false
FOUND_ROOT_DIR=false

# Attempt 1: Try to find the root dir by looking at the current executing script's directory
SCRIPT_DIR=$(dirname "$0")
ROOT_DIR=$(readlink -f "$SCRIPT_DIR/..")
if check_is_correct_root_dir "$ROOT_DIR"; then
    FOUND_ROOT_DIR=true
else
    echo "... could not find the root directory by going up from SCRIPT_DIR"
    echo "... trying to find the root directory by going up from PWD"
fi

# Attempt 2: Try to find the root dir by looking at PWD
if ! $FOUND_ROOT_DIR; then
    ROOT_DIR=$(readlink -f "$PWD")
    if check_is_correct_root_dir "$ROOT_DIR"; then
        FOUND_ROOT_DIR=true
    else
        echo "... could not find the root directory from PWD"
        echo "... trying to find the root directory from 'BASH_SOURCE' (only works in bash)"
    fi
fi

# Attempt 3: Try to find the root via BASH_SOURCE (works only in bash)
if ! $FOUND_ROOT_DIR && [ -n "$BASH_VERSION" ]; then
    SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
    ROOT_DIR=$(readlink -f "$SCRIPT_DIR/..")
    if check_is_correct_root_dir "$ROOT_DIR"; then
        FOUND_ROOT_DIR=true
    else
        echo "... could not find the root directory from 'BASH_SOURCE' (only works in bash)"
        echo "Options exhausted. If you are 'source'ing this script from another script, you must cd into the cifutils directory and run the script from there."
        exit 1
    fi
fi

# Check if a valid root directory was found
if ! $FOUND_ROOT_DIR; then
    echo "ERROR! Could not determine the correct root directory."
    exit 1
else
    echo "Successfully found the cifutils root directory: $ROOT_DIR"
    echo 
fi

echo "Checking if $ROOT_DIR needs to be added to PYTHONPATH..."
if ! echo $PYTHONPATH | grep -q "$ROOT_DIR/src"; then
    echo "... adding $ROOT_DIR/src to your PYTHONPATH"
    export PYTHONPATH=$ROOT_DIR/src:$PYTHONPATH
else
    echo "... it's already in PYTHONPATH"
fi

# check if we are at the IPD
echo
echo "Verifying that you are on the IPD cluster"
IPD_FILE="/software/containers/versions/rf_diffusion_aa/ipd.txt"
if [ -f $IPD_FILE ]; then
    echo "... you are on the IPD cluster!"
else
    echo "ERROR! You appear not to be on the IPD cluster. The setup script is only available on the IPD cluster."
    exit 1
fi

# Ensure a .env file exists, otherwise create it
if [ ! -f .env ]; then
    echo "Creating a .env file"
    touch .env
fi

# Read the .env file and check if the relevant 'env' variables are already set
echo "Adding CCD MIRROR to your environment variables"
export CCD_MIRROR_PATH=$(readlink -f $ROOT_DIR/.ipd/ccd)
echo "... CCD_MIRROR_PATH='$CCD_MIRROR_PATH'"
check_and_set_env_var "CCD_MIRROR_PATH" "$CCD_MIRROR_PATH"

echo
echo "Adding PDB MIRROR to your environment variables"
export PDB_MIRROR_PATH=$(readlink -f $ROOT_DIR/.ipd/pdb)
echo "... PDB_MIRROR_PATH='$PDB_MIRROR_PATH'"
check_and_set_env_var "PDB_MIRROR_PATH" "$PDB_MIRROR_PATH"

echo 
echo "Adding AF2FB_PATH to your environment variables"
export AF2FB_PATH=$(readlink -f $ROOT_DIR/.ipd/af2_distillation_fb)
echo "... AF2FB_PATH='$AF2FB_PATH'"
check_and_set_env_var "AF2FB_PATH" "$AF2FB_PATH"

echo "Your .env file is:"
echo "================================================"
cat .env
echo "================================================"
echo
echo "Done!"
