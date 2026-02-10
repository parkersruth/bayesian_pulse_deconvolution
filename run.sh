
cmd=(
    snakemake
    --use-conda
    --quiet rules
    # --rerun-triggers mtime
    --cores 1
    "$@" # *args **kwargs
)

# print command
echo
printf "%q " "${cmd[@]}"
echo
echo

# execute command
"${cmd[@]}"
