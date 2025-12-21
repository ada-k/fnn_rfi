# AOFlagger Docker build

This folder provides a reproducible Docker environment to build and run AOFlagger without modifying the host.

Quick steps

1. Build the image:

```bash
cd tools/aoflagger_docker
./build_and_run.sh build
```

2. Run the container with your data folder mounted at `/data`:

```bash
./build_and_run.sh run /path/to/your/observations
# inside container: aoflagger (or look for installed binaries)
```

Notes
- The Dockerfile builds `casacore` first (installed to the container) then `AOFlagger` from the official repo.
- Builds can be large and take several minutes depending on CPU.
- Use the container to run AOFlagger on measurement sets and then copy produced mask files back to the host path you mounted.
