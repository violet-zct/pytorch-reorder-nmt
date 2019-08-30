saved_exps -> /projects/efs/users/chuntinz/syn_saved_exps/

chuntinz@ec2-54-244-193-17.us-west-2.compute.amazonaws.com:/projects/efs/users/chuntinz/syn_saved_exps/

python ./subword_nmt/learn_bpe.py -s 16000 < ./ug_indomain_datasets/train-dev.low.tok.ug > ./ug_codes/train-dev.low.tok.ug.codes
python ./subword_nmt/apply_bpe.py -c ./ug_codes/train-dev.low.tok.ug.codes < ./ug_indomain_datasets/train.low.tok.ug > ./ug_indomain_datasets/train.low.tok.ug.bpe
python ./subword_nmt/apply_bpe.py -c ./ug_codes/train-dev.low.tok.ug.codes < ./ug_indomain_datasets/dev.ug.lower.tok > ./ug_indomain_datasets/dev.ug.lower.tok.lr-bpe
python ./subword_nmt/apply_bpe.py -c ./ug_codes/train-dev.low.tok.ug.codes < ./ug_indomain_datasets/test.ug.lower.tok > ./ug_indomain_datasets/test.ug.lower.tok.lr-bpe

python ./subword_nmt/learn_bpe.py -s 16000 < ./ug_indomain_datasets/train-dev.low.tok.en > ./ug_codes/train-dev.low.tok.en.codes
python ./subword_nmt/apply_bpe.py -c ./ug_codes/train-dev.low.tok.en.codes < ./ug_indomain_datasets/train.low.tok.en > ./ug_indomain_datasets/train.low.tok.en.bpe

python cat_files.py train-dev.low.tok.en train.low.tok.en dev.en.lower.tok.ref.2
python cat_files.py train-dev.low.tok.ug train.low.tok.ug dev.ug.lower.tok
python cat_files.py all.low.tok.en train.low.tok.en aug.low.tok.ori dev.en.lower.tok.ref.2 
python cat_files.py all.low.tok.ug train.low.tok.ug aug.low.tok.ori.tran dev.ug.lower.tok

python cat_files.py train-aug.low.tok.ug.ori train.low.tok.ug aug.low.tok.ori.tran
python cat_files.py train-aug.low.tok.ug.hf train.low.tok.ug aug.low.tok.hf.tran
python cat_files.py train-aug.low.tok.ug.hf.novar train.low.tok.ug aug.low.tok.hf.novar.tran
python cat_files.py train-aug.low.tok.en train.low.tok.en aug.low.tok.ori

python cat_files.py train.low.tok.en.lex train.low.tok.en en.lexicon
python cat_files.py train.low.tok.ug.lex train.low.tok.ug ug.lexicon
python cat_files.py train-dev.low.tok.ug.lex train-dev.low.tok.ug ug.lexicon
python cat_files.py train-dev.low.tok.en.lex train-dev.low.tok.en en.lexicon
python cat_files.py train-aug.low.tok.en.lex train-aug.low.tok.en en.lexicon 
python cat_files.py train-aug.low.tok.ug.hf.lex train-aug.low.tok.ug.hf ug.lexicon
python cat_files.py train-aug.low.tok.ug.hf.novar.lex train-aug.low.tok.ug.hf.novar ug.lexicon
python cat_files.py train-aug.low.tok.ug.ori.lex train-aug.low.tok.ug.ori ug.lexicon
python cat_files.py all.low.tok.en.lex all.low.tok.en en.lexicon
python cat_files.py all.low.tok.ug.lex all.low.tok.ug ug.lexicon

python ./subword_nmt/learn_bpe.py -s 25000 < ./ug_indomain_datasets/all.low.tok.en > ./ug_codes/all.low.tok.en.codes
python ./subword_nmt/apply_bpe.py -c ./ug_codes/all.low.tok.en.codes < ./ug_indomain_datasets/train-aug.low.tok.en > ./ug_indomain_datasets/train-aug.low.tok.en.bpe
python ./subword_nmt/apply_bpe.py -c ./ug_codes/all.low.tok.en.codes < ./ug_indomain_datasets/en.lexicon > ./ug_indomain_datasets/en.lexicon.bpe
python ./subword_nmt/apply_bpe.py -c ./ug_codes/all.low.tok.en.codes < ./ug_indomain_datasets/aug.low.tok.ori > ./ug_indomain_datasets/aug.low.tok.ori.bpe

python ./subword_nmt/learn_bpe.py -s 25000 < ./ug_indomain_datasets/all.low.tok.ug > ./ug_codes/all.low.tok.ug.codes
rem python ./subword_nmt/apply_bpe.py -c ./ug_codes/all.low.tok.ug.codes < ./ug_indomain_datasets/aug.low.tok.hf_novar.tran > ./ug_indomain_datasets/aug.low.tok.hf_novar.tran.bpe
rem python ./subword_nmt/apply_bpe.py -c ./ug_codes/all.low.tok.ug.codes < ./ug_indomain_datasets/aug.low.tok.ori.tran > ./ug_indomain_datasets/aug.low.tok.ori.tran.bpe
python ./subword_nmt/apply_bpe.py -c ./ug_codes/all.low.tok.ug.codes < ./ug_indomain_datasets/train-aug.low.tok.ug.ori > ./ug_indomain_datasets/train-aug.low.tok.ug.ori.bpe
python ./subword_nmt/apply_bpe.py -c ./ug_codes/all.low.tok.ug.codes < ./ug_indomain_datasets/train-aug.low.tok.ug.hf > ./ug_indomain_datasets/train-aug.low.tok.ug.hf.bpe
python ./subword_nmt/apply_bpe.py -c ./ug_codes/all.low.tok.ug.codes < ./ug_indomain_datasets/train-aug.low.tok.ug.hf.novar > ./ug_indomain_datasets/train-aug.low.tok.ug.hf.novar.bpe
python ./subword_nmt/apply_bpe.py -c ./ug_codes/all.low.tok.ug.codes < ./ug_indomain_datasets/dev.ug.lower.tok > ./ug_indomain_datasets/dev.ug.lower.tok.aug-bpe
python ./subword_nmt/apply_bpe.py -c ./ug_codes/all.low.tok.ug.codes < ./ug_indomain_datasets/test.ug.lower.tok > ./ug_indomain_datasets/test.ug.lower.tok.aug-bpe
python ./subword_nmt/apply_bpe.py -c ./ug_codes/all.low.tok.ug.codes < ./ug_indomain_datasets/ug.lexicon > ./ug_indomain_datasets/ug.lexicon.bpe

scp xuezhem@rabat.sp.cs.cmu.edu:/home/xuezhem/ct/reorder-nmt/ug_prep/core_ug_data/aug.low.tok.ori.selected ./aug.low.tok.ori
scp xuezhem@rabat.sp.cs.cmu.edu:/home/xuezhem/ct/reorder-nmt/ug_prep/core_ug_data/aug.low.tok.*tran ./

perl ../Datasets/multi-bleu.perl ../data/uyghur/test.en.lower.tok.ref.0 ../data/uyghur/test.en.lower.tok.ref.1 ../data/uyghur/test.en.lower.tok.ref.2 ../data/uyghur/test.en.lower.tok.ref.3 < 

python3 compare_mt.py /home/chuntinz/ug_reorder/test.en.lower.tok.ref.0 /home/chuntinz/ug_reorder/ae.tran /home/chuntinz/ug_reorder/hf.tran 

perl ../../Datasets/multi-bleu.perl ./test.en.lower.tok.ref.0 ./test.en.lower.tok.ref.1 ./test.en.lower.tok.ref.2 ./test.en.lower.tok.ref.3 < 

