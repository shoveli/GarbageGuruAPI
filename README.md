# garbage-guru-api
An API written in Python with Flask for classifying and recycling garbage objects using an AI model. It was created for the Applied AI course in Informatics Engineering - UNLaM 2024. 

The API communicates with the [GarbageGuru Android app](https://github.com/joniaranguri/garbage-guru-android) as outlined in the architecture design:

![Garbage Guru Overall Architecture](images/garbage-guru-architecture-overall-architechture.png)


## Running the API
1. **Build the Docker image**:

   ```bash
   docker build -t flask-tensorflow-app .
   ```

2. **Run the Docker container**:

   ```bash
   docker run -p 5000:5000 flask-tensorflow-app
   ```

   This will run the Flask app inside a Docker container, exposing it on port 5000.