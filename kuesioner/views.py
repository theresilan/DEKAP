from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.http import JsonResponse
import json
import os
import os
import pandas as pd
import pickle


@api_view(['GET', 'POST'])
def kuesioner(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            print(data);
            # Extract key1_value from the received data
            os.system('python filename.py')
            df = pd.DataFrame([data])

            print(df.columns)

            with open("kuesioner/randomf.pkl", "rb") as file:
                loadedmodel = pickle.load(file)

            test_predictions = loadedmodel.predict(df)

            print(test_predictions)

            # Perform actions based on received data
            test_predictions_list = test_predictions.tolist()

            # Prepare response data
            response_data = {
                'predictions': test_predictions_list
            }

            # Return JSON response
            return JsonResponse(response_data)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    elif request.method == 'GET':
        # Handle GET request, if needed
        return JsonResponse({'message': 'GET request received'}, status=200)
    else:
        return JsonResponse({'error': 'Method Not Allowed'}, status=405)
