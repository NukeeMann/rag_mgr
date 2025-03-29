# Copyright (c) Opendatalab. All rights reserved.
import os

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

# args
pdf_path = "pdf_files/book_Nauka_o_Cierpieniu.pdf" 
name_without_suff = pdf_path.split('/')[-1].split(".")[0]

# prepare env
local_image_dir, local_md_dir = f"{name_without_suff}/images", f"{name_without_suff}"
image_dir = str(os.path.basename(local_image_dir))
print(image_dir)

os.makedirs(local_image_dir, exist_ok=True)

image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
    local_md_dir
)

# Read the pdf content
reader1 = FileBasedDataReader("")
pdf_bytes = reader1.read(pdf_path)  

# Create Dataset Instance
ds = PymuDocDataset(pdf_bytes, lang="pl")

# inference
infer_result = ds.apply(doc_analyze, ocr=False, lang="pl")

# pipeline
pipe_result = infer_result.pipe_txt_mode(image_writer)

# dump markdown
pipe_result.dump_md(md_writer, f"{name_without_suff}.txt", image_dir)
