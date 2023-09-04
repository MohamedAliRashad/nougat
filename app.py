"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import sys
from functools import partial
from http import HTTPStatus
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pathlib import Path
import hashlib
from fastapi.middleware.cors import CORSMiddleware
import fitz
import torch
from nougat import NougatModel
from nougat.postprocessing import markdown_compatible, close_envs
from nougat.utils.dataset import ImageDataset
from nougat.utils.checkpoint import get_checkpoint
from nougat.dataset.rasterize import rasterize_paper
from tqdm import tqdm

SAVE_DIR = Path("./pdfs")
BATCHSIZE = os.environ.get("NOUGAT_BATCHSIZE", 6)
NOUGAT_CHECKPOINT = get_checkpoint()
if NOUGAT_CHECKPOINT is None:
    print(
        "Set environment variable 'NOUGAT_CHECKPOINT' with a path to the model checkpoint!."
    )
    sys.exit(1)

app = FastAPI(title="LaTeX-OCR API")
origins = ["http://localhost", "http://127.0.0.1"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = None


@app.on_event("startup")
async def load_model(
    checkpoint: str = NOUGAT_CHECKPOINT,
):
    global model
    if model is None:
        model = NougatModel.from_pretrained(checkpoint).to(torch.bfloat16)
        if torch.cuda.is_available():
            model.to("cuda")
        model.eval()


@app.get("/")
def root():
    """Health check."""
    response = {
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...), start: int = None, stop: int = None
) -> str:
    """
    Perform predictions on a PDF document and return the extracted text in Markdown format.

    Args:
        file (UploadFile): The uploaded PDF file to process.
        start (int, optional): The starting page number for prediction.
        stop (int, optional): The ending page number for prediction.

    Returns:
        str: The extracted text in Markdown format.
    """
    pdfbin = file.file.read()
    pdf = fitz.open("pdf", pdfbin)

    if start is not None and stop is not None:
        pages = list(range(start - 1, stop))
    else:
        pages = list(range(len(pdf)))
    predictions = [""] * len(pages)
    dellist = []
    compute_pages = pages.copy()
    for el in dellist:
        compute_pages.remove(el)
    images = rasterize_paper(pdf, pages=compute_pages)
    global model

    dataset = ImageDataset(
        images,
        partial(model.encoder.prepare_input, random_padding=False),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCHSIZE,
        pin_memory=True,
        shuffle=False,
    )

    for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        if sample is None:
            continue
        model_output = model.inference(image_tensors=sample)
        for j, output in enumerate(model_output["predictions"]):
            if model_output["repeats"][j] is not None:
                if model_output["repeats"][j] > 0:
                    disclaimer = "\n\n+++ ==WARNING: Truncated because of repetitions==\n%s\n+++\n\n"
                else:
                    disclaimer = (
                        "\n\n+++ ==ERROR: No output for this page==\n%s\n+++\n\n"
                    )
                rest = close_envs(model_output["repetitions"][j]).strip()
                if len(rest) > 0:
                    disclaimer = disclaimer % rest
                else:
                    disclaimer = ""
            else:
                disclaimer = ""

            predictions[pages.index(compute_pages[idx * BATCHSIZE + j])] = markdown_compatible(output)

    final = "".join(predictions).strip()
    return final


def main():
    import uvicorn

    uvicorn.run("app:app", port=8503)


if __name__ == "__main__":
    main()
