# oculomotor
WBAI 2018 Hackathon Oculomotor Project Repository.

## Usage
### 1. Clone this repository.
```
$ git clone --recursive https://github.com/wbap/oculomotor
```

### 2. Build the base docker image.
```
# 2. Build the base docker image.
$ docker build -t wbap/oculomotor-base ./base
```

### 3. Edit the files under `appliation/functions`.

### 4. Build the server docker image.
```
$ docker build -t wbap/oculomotor-server .
```

### 5. Run the server docker image.
```
$ docker run -it -p 8080:80 wbap/oculomotor-server
```

### 6. Run the client script.
```
$ python client.py
```
