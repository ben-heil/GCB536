# This script runs on the same data as all the others in the directory, but
# scvis requires you to convert it to a tsv
scvis train --data_matrix_file ./data/data_copy.tsv \
    --out_dir ./output/scvisout \
    --data_label_file ./data/labels_copy.tsv \
    --verbose \
    --verbose_interval 50
