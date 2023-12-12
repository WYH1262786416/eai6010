from fastai.text.all import *
from flask import Flask, request, Response
import functools
from google.cloud import storage
import json
import os


app = Flask( __name__ )
storage_client = storage.Client( )


@app.route( "/classify", methods = [ "POST" ] ) # when I post to the {base}/classify URL I call this function
def classify_article( ):
    """
    Classifies a news article given its "headline" and "shortDescription".
    """

    #   Load the classifier
    classifier = _load_classifier( )

    #   Classify the input normally
    request_json = request.get_json( )
    headline = request_json.get( "headline", None )
    if headline is not None:

        #   Make the prediction
        content = f"{headline}"
        prediction = classifier.predict( content )[0]

        #   Build the response
        response_json = { "predictedClass": prediction }
        response_json_string = json.dumps( response_json )
        response = Response( response_json_string,  mimetype='application/json' )
        return response

    #   We are missing content, complain to the user
    else:
        response_json = { "message": "headline are required to make a class prediction" }
        response_json_string = json.dumps( response_json )
        response = Response( response_json_string,  mimetype='application/json' )
        return response


@functools.lru_cache( maxsize = 1 )
def _load_classifier():
    """ Loads the classifier from the local file system. """
    classifier = load_learner("model.pkl") 
    return classifier


if __name__ == "__main__":
    app.run( debug = True, host = "0.0.0.0", port = int( os.environ.get( "PORT", 5000) ) )

