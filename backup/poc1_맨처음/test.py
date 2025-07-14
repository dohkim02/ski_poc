from _preprocess import excel_to_txt_chunks_parallel

file_path = "../data/preprocessed_data.xlsx"
chunks = excel_to_txt_chunks_parallel(file_path)

import pdb

pdb.set_trace()
