from fastapi import FastAPI

app = FastAPI(title="FastAPI Blog")


@app.get("/")
def read_root():
    return {"message": "FastAPI Blog is running"}
