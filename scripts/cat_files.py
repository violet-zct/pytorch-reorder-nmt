import io
import random
import numpy as np

dir_path = "../data/aspec/"

jej_path = dir_path + "aspec.low.train.ja-en.ja."

tran_hf_path = dir_path + "aspec.low.train.hf.tran."
tran_ae_path = dir_path + "aspec.low.train.en.tran."

output_ja_hf = dir_path + "aspec.low.train.combine.hf.ja."
output_ja_en = dir_path + "aspec.low.train.combine.en.ja."

jee_path = dir_path + "aspec.low.train.ja-en.en."
mono_en_path = dir_path + "aspec.low.train.hf-en.en."
output_en = dir_path + "aspec.low.train.combine.en."

labels = ["3000", "6000", "10000", "20000"]


def read_lines(path, repeat=1):
    lines = []
    with io.open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            lines.append(line.strip())
    lines *= repeat
    return lines


def combine(p1, p2, r1=1, r2=1):
    lines1 = read_lines(p1, r1)
    lines2 = read_lines(p2, r2)
    lines = lines1 + lines2
    return lines


def write_out(lines, op):
    with io.open(op, "w", encoding="utf-8") as fout:
        for line in lines:
            fout.write(line + "\n")


def shuffle_lines(lines_1, lines_2, lines_3, op1, op2, op3):
    tot = len(lines_1)
    assert len(lines_1) == len(lines_2) and len(lines_1) == len(lines_3)
    indx = np.random.permutation(np.arange(tot))
    lines_1 = np.array(lines_1)
    lines_2 = np.array(lines_2)
    lines_3 = np.array(lines_3)
    lines_1 = list(lines_1[indx])
    lines_2 = list(lines_2[indx])
    lines_3 = list(lines_3[indx])
    write_out(lines_1, op1)
    write_out(lines_2, op2)
    write_out(lines_3, op3)

repeat = 5
str_repeat = "." + str(repeat)

repeat = 1
str_repeat = ""

for l in labels:
    tjej_path = jej_path + l

    ttran_hf_path = tran_hf_path + l
    toutput_ja_hf = output_ja_hf + l + str_repeat
    ttran_ae_path = tran_ae_path + l
    toutput_ja_en = output_ja_en + l + str_repeat

    tjee_path = jee_path + l
    tmono_en_path = mono_en_path + l
    toutput_en = output_en + l + str_repeat

    ja_hf = combine(tjej_path, ttran_hf_path, repeat)
    ja_en = combine(tjej_path, ttran_ae_path, repeat)
    en = combine(tjee_path, tmono_en_path, repeat)

    shuffle_lines(ja_hf, en, ja_en, toutput_ja_hf, toutput_en, toutput_ja_en)
