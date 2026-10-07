pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    environment {
        MLFLOW_TRACKING_URI = "${env.MLFLOW_TRACKING_URI ?: 'sqlite:///mlflow.db'}"
        MLFLOW_MODEL_NAME = "${env.MLFLOW_MODEL_NAME ?: 'titanic-survival-random-forest'}"
        MIN_TEST_ACCURACY = "${env.MIN_TEST_ACCURACY ?: '0.75'}"
        VENV = '.venv'
    }

    stages {
        stage('Install dependencies') {
            steps {
                sh '''
                    set -eu
                    python3 -m venv "${VENV}"
                    . "${VENV}/bin/activate"
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Train and register model') {
            steps {
                sh '''
                    set -eu
                    . "${VENV}/bin/activate"
                    python train.py
                '''
            }
        }
    }

    post {
        success {
            echo "Registered ${MLFLOW_MODEL_NAME} in ${MLFLOW_TRACKING_URI} and set the production alias."
        }
        failure {
            echo "Training or registration failed. The production alias was not updated."
        }
    }
}
