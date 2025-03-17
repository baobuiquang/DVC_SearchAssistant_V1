from rapidfuzz import fuzz as fuzz_fuzz
from rapidfuzz import process as fuzz_process
import numpy as np
import csv

with open('url/cache', mode='r', newline='', encoding='utf-8') as f:
    csv_reader = csv.DictReader(f)
    data = [row for row in csv_reader]
    for i in range(len(data)):
        data[i]["Tên thủ tục"] = data[i]["Tên thủ tục"].upper()

def semanticsearch(text, top_limit):
    text = text.upper()
    print(text)
    res = []
    for e in fuzz_process.extract(text, [e["Tên thủ tục"] for e in data], scorer=fuzz_fuzz.WRatio, limit=top_limit):
        print(e)
        res.append(data[e[2]]["Mã chuẩn"])
    return res

# ====================================================================================================

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# > http://localhost:8888/?top=1&text=phuc%20khao%20tot%20nghiep
@app.get("/")
def ohyeah(top: int=1, text: str=""):
    if text == "":
        return "Semantic search API is working!"
    else:
        return {
            "result": semanticsearch(text, top)
        }

uvicorn.run(
    app, 
    host = "127.0.0.1",
    port = 8888,
    log_config = None,
)