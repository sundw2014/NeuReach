echo '' > test.txt
for seed in {10..19};
do
    python3 scripts/eval.py --no_cuda --pretrained $2/checkpoint.pth.tar --system $1 --seed $seed --output test.txt 1>/dev/null 2>&1
done
python3 scripts/parse_res.py test.txt $2
