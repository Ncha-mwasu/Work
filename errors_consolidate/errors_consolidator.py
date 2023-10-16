import json
import os

from collections import OrderedDict



def json_orchestrate(predicator_length, forecast_length, category, errors_dict):
    '''
    For example:
    json_orchestrate(10, 5, 'arima', {'mae': 0.32, 'mse': 0.047}) ;

    In the above call, json_orchestrator will add an entry or make updates to an existing entry in the errors_parallel.json file.

    Arguments Elaboration:
    Using 10 records from history to make 5 consecutive future forecasts. The mechanism used for forecasting is 'arima' with  mean 
    absolute error and mean squared error = (0.32, 0.047) respectively.
    '''
    
    if not (isinstance(predicator_length, int) and isinstance(forecast_length, int)):
        raise Exception('predicator_length and forecast_length are integers')
    elif category not in ['arima', 'markov']:
        raise Exception('category must be one of (arima | markov)')
    elif not (isinstance(errors_dict, OrderedDict) or isinstance(errors_dict, dict)):
        raise Exception('errors_dict must be a dictionary or ordered dictionary')
    elif len(errors_dict) == 0:
        raise Exception('errors_dict cannot be empty')
    elif not len(errors_dict) <= 2:
        raise Exception('only 2 errors allowed')
    elif not set(errors_dict).issubset({'mae', 'mse'}):
        raise Exception('errors must be one or both of (mae | mse)')
        
    this_errors_category = category
    this_errors_space = f'{predicator_length}p{forecast_length}'
    this_errors_dict = OrderedDict(errors_dict)

    # The json file path
    errors_file_path = 'errors_consolidate\errors_parallel.json'

    # Create the file if it doesn't exist
    if not os.path.exists(errors_file_path):
        temp_file = open(errors_file_path, 'w')
        json.dump({}, temp_file)
        temp_file.close()

    # Load data from the file
    errors_file = open(errors_file_path, 'r')
    this_data = OrderedDict(json.load(errors_file))
    errors_file.close()

    '''
    Orchestrate
    '''
    if this_errors_space not in this_data: # Add new space
        new_space_dict = OrderedDict()
        new_space_dict[this_errors_space] = OrderedDict({category: this_errors_dict})
        this_data.update(new_space_dict)
    else: # Update the space
        new_errors_dict = OrderedDict()
        this_data[this_errors_space].update(OrderedDict({category: this_errors_dict}))


    # Dump new data to the file
    with open(errors_file_path, 'w') as errors_file:
        json.dump(this_data, errors_file)
