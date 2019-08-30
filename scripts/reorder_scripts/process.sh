P=$1

# 1. split sentences and split file into ${P} splits for multi-processes
python split_filt.py train.en.100 ${P}

# 2. run enju parser
bash parse_multi_process.sh ${P}

# 3. combine multiple outputs from multi-processes into the full parsed Engligh data
cd ./split_files/
python ../combine_parse.py ${P}

# 4. run head finalization over parsed datas
cd ../

# Special handling quote mark: first replace &quot; with _QUOT_ for both full.xml.parse and the target English file
cat ./split_files/full.xml.parse | sed 's/&quot;/_QUOT_/g' > ./split_files/full.xml.quot.replace.parse
cat train.en.100.tok | sed 's/&quot;/_QUOT_/g' > train.en.100.qr.tok

# Check
grep -o -w "&quot;" ./split_files/full.xml.quot.replace.parse | wc -w
grep -w -w "&quot;" train.en.100.qr.tok | wc -w

#python head5.py < ./split_files/full.xml.parse > ./full.hf
python head5.py < ./split_files/full.xml.quot.replace.parse > ./full.hf

# output: recovered.hf
# 5. recover the orighinal training data based on the split.sent.num file outputted by step 1
python recover_dataset_from_hf.py full.hf sent.split.num

# 6. lower case the hf file
tr '[:upper:]' '[:lower:]' < recovered.hf.aspec > recovered.hf.aspec.low
# replace _QUOT_ in _quot_
# 7. split train into supervised and semi-supervised
python fetch_lines.py train.ja.100.tok.low train.en.100.tok.low recovered.hf.aspec.low aspec 30

# check unk words
# eos, bos, unk, word embedding assign the same