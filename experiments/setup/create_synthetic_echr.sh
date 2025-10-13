
args=(
    -m src.synthetic.generation
    --rights "echr"
    # --overwrite
    # --test
)

python "${args[@]}"
