import json
import glob

OP_DIR = 'openpose_out'
OUT_CSV = 'test-annotation.csv'

# CSV_IN_OP[CSV] = OP
CSV_IN_OP = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

l_json = glob.glob(f'{OP_DIR}/*.json')
annots = {}

for f_path in l_json:
    with open(f_path) as f:
        j_annot = json.load(f)
    j_annot = j_annot['people'][0]['pose_keypoints_2d']
    xs = [round(j_annot[3 * i]) if j_annot[3 * i + 2] else -1 for i in CSV_IN_OP]
    ys = [round(j_annot[3 * i + 1]) if j_annot[3 * i + 2] else -1 for i in CSV_IN_OP]
    print(f_path)
    print(xs)
    print(ys)
