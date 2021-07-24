# Test task

### Supported Python versions 3.7-3.9

### Clone and install dependencies:
```
pip install -r requirements.txt
```
### Cd into the repo folder and download the test video file
```
wget https://drive.google.com/uc?export=download&id=1xGVHqoxUR6dG7npRJgsAzWdfFYXwSonq
```


### Run the detection script
```
python detectVehicles.py video.mkv out.mkv
```
*if you run it locally the output will be shown at runtime. Docker image will only output the framerates and the file itself

### Or pull and run the docker image
```
docker pull rumblingturtle/test
docker run rumblingturtle/test
```

### Or build it
```
docker build .
```