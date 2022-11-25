from fastapi import FastAPI

app = FastAPI()

@app.get('/asd')
async def root():
    return {"message": "Hello worldasd!"}

@app.get('/')
async def root():
    return {"message": "Hello world!"}


