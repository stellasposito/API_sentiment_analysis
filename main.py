from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from sentiment_analysis import sentiment_analyzer

app = FastAPI()

@app.get("/analyze_sentiment")
def analyze_sentiment(product_id: str = Query(...)):
    try:
        result = sentiment_analyzer(product_id)
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

