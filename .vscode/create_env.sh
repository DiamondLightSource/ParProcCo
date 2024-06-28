#! /bin/bash

current_directory="$(dirname "$0")"
env_file="$current_directory/.env"

if [ -f "$env_file" ]; then
    # shellcheck source=.env
    source "$env_file"
    rm "$env_file"
fi

# Update the token
SLURM_JWT="$(cat ~/.ssh/slurm.tkn)"

echo "Creating $env_file with the following contents:"
printf "SLURM_REST_URL=%s\nSLURM_PARTITION=%s\nSLURM_JWT=%s\n" "$SLURM_REST_URL" "$SLURM_PARTITION" "$SLURM_JWT" | tee "$env_file"
