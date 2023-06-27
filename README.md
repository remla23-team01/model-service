[![Release](https://github.com/remla23-team01/model-service/actions/workflows/release.yml/badge.svg)](https://github.com/remla23-team01/model-service/actions/workflows/release.yml)
# model-service
Small backend to do model predictions with.

To run the model use the command:
```
pip install -r requirements.txt; python ./app.py
```

Or to run the service in a Docker container:

```
docker run --rm -p 8080:8080 ghcr.io/remla23-team01/model-service:latest
```

To find all endpoints use the [APIDocs](http://localhost:8080/apidocs/)
