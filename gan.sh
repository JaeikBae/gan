docker build -t gan .
xhost +
nvidia-docker run \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v .:/workspace \
    --rm -it \
    gan \
    /bin/bash
