

Run docker container
```
docker build . --tag people_counter

docker run --rm -it --init --gpus=all --ipc=host --user="$(id -u):$(id -g)" --volume="$PWD:/app" people_counter
```

Run face detection algorithm
```
python face_detection.py <path to the video> <path to the config> <path to the weights>
```