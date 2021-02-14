

Run docker container
```
cd FairMOT/build
docker build --build-arg IMAGE_NAME=nvidia/cuda --tag fair_mot .
cd ../..

docker build --tag people_counter .

docker run --rm -it --init --gpus=all --ipc=host --user="$(id -u):$(id -g)" --volume="$PWD:/app" people_counter
```
