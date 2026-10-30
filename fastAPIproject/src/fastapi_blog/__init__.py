def main() -> None:
    import uvicorn

    uvicorn.run("fastapi_blog.main:app", reload=True)


if __name__ == "__main__":
    main()
 