import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        timeout_keep_alive=3600,  # Increase timeout for long processing
    )
