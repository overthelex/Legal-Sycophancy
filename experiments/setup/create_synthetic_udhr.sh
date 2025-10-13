
args=(
    -m src.synthetic.generation
    --rights "udhr"
    # --overwrite
    # --test
)

python "${args[@]}"
