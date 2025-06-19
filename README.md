
```bash
docker build -t opticalflow .
```

```bash
# Basic profiling (analysis-only, fast)
docker run --rm -v $(pwd)/videos:/app/videos optical-flow-headless:latest \
  python dense_profiling.py --video /app/videos/input.mp4 --max-frames 50

# generate video outputs
docker run --rm -v $(pwd)/videos:/app/videos -v$(pwd)/output:/app/output opticaflow python dense_headless.py --video /app/videos/input.mp4 --save-videos

