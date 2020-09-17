#!/usr/bin/env sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CUR_DIR="$(realpath pwd)"
DATA_DIR="$(realpath $SCRIPT_DIR/../data/squad_sqg)"
cd DATA_DIR

cat squad11_sqg.train.tsv | cut -f1 > source
cat squad11_sqg.train.tsv | cut -f2 > target

head -n 85000 source > train.source
head -n 85000 target > train.target

tail -n 2599 squad11_sqg.train.tsv | awk '!seen[$1]++' > valtest

split -n2 valtest

cat xaa | cut -f1 > valid.source
cat xaa | cut -f2 > valid.target
cat xab | cut -f1 > test.source
cat xab | cut -f2 > test.target

rm xaa xab source target

cd ..
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR/..
TASK=data/squad_sqg
for SPLIT in train val
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$TASK/$SPLIT.$LANG" \
    --outputs "$TASK/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/val.bpe" \
  --destdir "${TASK}-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
