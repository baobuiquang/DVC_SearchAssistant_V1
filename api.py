from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ====================================================================================================

from app import fn_ohyeahhhhhhhhhhhhhhhhhhhhhhhhhh_non_streaming

# ====================================================================================================

# > http://localhost:5002/?input_text=T%C3%B4i%20mu%E1%BB%91n%20kh%E1%BB%9Fi%20nghi%E1%BB%87p%20th%C3%AC%20c%E1%BA%A7n%20th%E1%BB%A7%20t%E1%BB%A5c%20n%C3%A0o?
@app.get("/")
def myapiyeah(input_text: str = ""):
    return fn_ohyeahhhhhhhhhhhhhhhhhhhhhhhhhh_non_streaming(message=input_text, history=[], sleeptime=0.0)

# ====================================================================================================
if __name__ == "__main__":
    uvicorn.run(app, host = "localhost", port = 5002)