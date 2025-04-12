from fastapi import FastAPI
from .routers import ragRouter

app = FastAPI(
    title='Bielik LLM',
    description='Bielik v2.3 Instruct',
    version='0.1',
)

app.include_router(ragRouter.router)
