from fastapi import FastAPI
from api.endpoints import router
import uvicorn

app = FastAPI(title="EORA Assistant API", version="1.0")

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


