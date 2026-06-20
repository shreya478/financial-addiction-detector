from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from api.deps import limiter, ml_cache
from api.routes import router

app = FastAPI(title="Financial Addiction Detector API", version="1.0.0")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(router)

@app.on_event("startup")
async def startup_event():
    # Model cache is a singleton, this forces it to load into memory on startup
    ml_cache.initialize()
    print("API successfully started and ML models loaded.")
