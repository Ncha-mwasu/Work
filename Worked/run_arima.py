import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import csv

from collections import OrderedDict

import config as cf
import errors_consolidate.errors_consolidator as ec



def stepwise_fit(train_data):
     # CHECK Bentropy MODEL
    from pmdarima import auto_arima
    import warnings
    warnings.filterwarnings("ignore") # Ignore harmless warnings
    return auto_arima(train_data, start_p = 0, start_q = 0,
                            max_p = 1, max_q = 1, m = 5,
                            start_P = 0, seasonal = False,
                            d = None, D = 1, trace = True,
                            error_action ='ignore',
                            suppress_warnings = True,
                            stepwise = True)

def forecast_accuracy(forecast, actual):
    # forecast, se, conf = self.result.forecast(len(self.test), alpha=0.05)  # 95% conf
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    mse = np.mean((forecast - actual)**2)       # MSE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    # return f'MAPE: {mape}/nME: {me}\nMAE: {mae}\nMPE: {mpe}\nRMSE: {rmse}' #\nCORR: {corr}\nMINMAX: {minmax}'
    return {'mape': mape, 'me': me, 'mae': mae, 'mpe': mpe,'mse': mse, 'rmse': rmse} #\nCORR: {corr}\nMINMAX: {minmax}'

def make_segmented_predictions(this_actual_csv_path, this_prediction_csv_path, step, consolidating_errors=False, predicted_as_next=False):
    import copy
    import math
    import re

    df = pd.read_csv(this_actual_csv_path, index_col='node')
    battoir_df = copy.deepcopy(df)
    if len(df.columns) < 2 * step:
        raise Exception('columns(rounds) length must be at least twice the step')
    # if len(df.columns) % step != 0:
    #     raise Exception('columns(rounds) length must be a multiple of step')
    
    # Node ids
    node_ids = battoir_df.index.tolist()

    # How many divisions along columns (segments) in this table
    segments_length = int(len(battoir_df.columns)/step)
    segments = OrderedDict()

    # Open CSV and make writer
    if not consolidating_errors:
        this_arima_predict_csv = open(this_prediction_csv_path.replace('.csv', f'_step-{step}.csv'), mode = 'w')
        this_arima_predict_csv_writer = csv.writer(this_arima_predict_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    else:
        this_arima_predict_csv = open(this_prediction_csv_path, mode='w')
        this_arima_predict_csv_writer = csv.writer(this_arima_predict_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Make segments divisions and add to segments OrderedDict
    segments_overflow_length = 0
    for i in range(1, segments_length + 1):
        if not (i == segments_length):
            segments[i] = battoir_df.iloc[:, (i-1) * step : i * step]
        else:
            segments[i] = battoir_df.iloc[:, (i-1) * step : ]
            this_segment_length = len(segments[i].columns)
            if this_segment_length > step:
                segments_overflow_length = this_segment_length


    # Predicted CSV header with indexes
    iheader = []
    # Predicted CSV rows to write
    rows = OrderedDict()
    # Initialize all rows with an empty list
    for i in range(0, len(df)):
        rows[f'row_{i}'] = []

    '''
    Accuracy Metrics
    '''


    # How many predictions are possible
    n_stages = segments_length - 1
    
    # Accuracy metrics
    average_forecast_accuracies = OrderedDict()
    # metrics = ['mape', 'me', 'mae', 'mpe', 'rmse']
    metrics = ['mae', 'mse']
    # Initialize all metrics with an empty list
    for i in range(1, n_stages + 1):
        average_forecast_accuracies[i] = [[] for i in metrics]

    # Accuracy metrics
    average_forecast_accuracies = OrderedDict()
    # metrics = ['mape', 'me', 'mae', 'mpe', 'rmse']
    metrics = ['mae', 'mse']
    # Initialize all metrics with an empty list
    for i in range(1, n_stages + 1):
        average_forecast_accuracies[i] = [[] for i in metrics]


    '''
    Accuracy Metrics
    '''

    # For each segment, 
    for k, v in segments.items():
        if k >= segments_length:
            break
        # Predict each row and add as (1/segments_length) to rows OrderedDict

        if predicted_as_next:
            genesis = {'segment': segments[1], 'segments_length': segments_length}
            def make_forecasts(genesis_seed):
                if genesis_seed['segments_length'] <= 0:
                    return []
                for _, crow in genesis_seed['segment'].iterrows():
                    fc, confint = stepwise_fit(crow.tolist()).predict(n_periods = step, return_conf_int=True, verbose=False)
                    fc_acc = forecast_accuracy(fc, segments[k + 1].iloc[i, :])

                    genesis_seed.update({'segments_length': genesis_seed['segments_length'] - 1})
                    return [{'fc': fc, 'fc_acc': fc_acc}] + make_forecasts(genesis_seed)
            fc = make_forecasts(genesis)

            i = 0
            for forecast in fc:
                for j, e in tuple(zip(range(0, len(metrics)), metrics)):
                    average_forecast_accuracies[k][j].append(forecast['fc_acc'][e])

                rows[f'row_{i}'].extend(forecast['fc'].tolist())
                i = i + 1

        
            # ###### Accuracy Metrics ##########
            # # Append all the metrics to average_forecast_accuracies OrderedDict
            # for j, e in tuple(zip(range(0, len(metrics)), metrics)):
            #     average_forecast_accuracies[k][j].append(fc_acc[e])
            # ###### Accuracy Metrics ##########
            
            # rows[f'row_{i}'].extend(fc.tolist())
            # i = i + 1

        else:
            i = 0
            for _, crow in segments[k].iterrows():
                if not k == (segments_length - 1):
                    # print('ttttttttttttttttttttt')
                    # print(segments_overflow_length)
                    fc, confint = stepwise_fit(crow.tolist()).predict(n_periods = step, return_conf_int=True, verbose=False)
                    fc_acc = forecast_accuracy(fc, segments[k + 1].iloc[i, :])
                elif not segments_overflow_length:
                    # print('aaaaaaaaaaaaaaaaa')
                    # print(segments_overflow_length)
                    # print()
                    fc, confint = stepwise_fit(crow.tolist()).predict(n_periods = step, return_conf_int=True, verbose=False)
                    # print(len(fc))
                    # print(segments)
                    # print(segments[k + 1])
                    # print(len(segments[k + 1].iloc[i, :]))
                    fc_acc = forecast_accuracy(fc, segments[k + 1].iloc[i, :])
                else:
                    fc, confint = stepwise_fit(crow.tolist()).predict(n_periods = segments_overflow_length, return_conf_int=True, verbose=False)
                    fc_acc = forecast_accuracy(fc, segments[k + 1].iloc[i, :])
                    


                ###### Accuracy Metrics ##########
                # Append all the metrics to average_forecast_accuracies OrderedDict
                for j, e in tuple(zip(range(0, len(metrics)), metrics)):
                    average_forecast_accuracies[k][j].append(fc_acc[e])
                ###### Accuracy Metrics ##########
                
                rows[f'row_{i}'].extend(fc.tolist())
                i = i + 1



        # Find (1/segments_length) iheader
        cheader = []
        for index, _ in segments[k].T.iterrows():
            pattern = r'\d+'
            match = re.search(pattern, index)
            cheader.append(int(match.group(0)))

        # Extend iheader List with cheader as (1/segments_length)
        iheader.extend(cheader)

    # Make the header
    header_start = int(np.min(iheader)) + step
    header_stop = int(np.max(iheader)) + step
    header = ['node_id'] + [f'round_{i}' for i in range(header_start, header_stop + 1)]

    # Write the rows header for predictions
    this_arima_predict_csv_writer.writerow(header)

    # Write all the rows
    i = 0 # node index start
    for k, v in rows.items():
        this_arima_predict_csv_writer.writerow([node_ids[i]] + rows[k])
        i = i + 1


    """
    Accuracy Metrics
    """

    # Write number of stages
    this_arima_predict_csv_writer.writerow(['no_of_stages', str(n_stages)])


    # # For each segment, 
    # for k, v in segments.items():
    #     if k >= segments_length:
    #         break
    #     # Add all metrics to their queue in every stage
    #     for _, crow in segments[k].iterrows():
    #         # Append all the metrics to average_forecast_accuracies OrderedDict
    #         for i, e in tuple(zip(range(0, len(metrics)), metrics)):
    #             average_forecast_accuracies[k][i].append(fc_acc[e])

    np_average_forecast_accuracies = OrderedDict()
    for k, v in average_forecast_accuracies.items():
        np_average_forecast_accuracies[k] = np.array(average_forecast_accuracies[k])
    # Conserving memory
    del average_forecast_accuracies

    # Aggregate all stages
    aggregate_average_forecast_accuracies = np.zeros((len(metrics), len(battoir_df)))
    for k, v in np_average_forecast_accuracies.items():
        aggregate_average_forecast_accuracies = aggregate_average_forecast_accuracies + np_average_forecast_accuracies[k]
    # Find average of each metric for every node
    avg_aggregate = aggregate_average_forecast_accuracies / n_stages
    # Transpose, easier to write to csv
    transposed_avg_aggregate = avg_aggregate.T
    # Average of all node metrics
    final_avg = np.mean(transposed_avg_aggregate, axis = 0)

    # Write accuracy metrics header
    this_arima_predict_csv_writer.writerow(['node_id'] + metrics)
    # Write accuracy metrics for every node
    for index, row in tuple(zip(range(0, len(transposed_avg_aggregate)), transposed_avg_aggregate)):
        this_arima_predict_csv_writer.writerow([node_ids[index]] + row.tolist())

    # Write final average of accuracy metrics header
    this_arima_predict_csv_writer.writerow(list(map(lambda x: 'avg_' + x, metrics)))
    # Write final average of accuracy metrics
    this_arima_predict_csv_writer.writerow(final_avg.tolist())



    # Close the predicted csv file
    this_arima_predict_csv.close()

    return final_avg.tolist()

def forced_deviation(row, mean):
    forced_deviation = 0
    for energy in row:
        forced_deviation = forced_deviation + (energy - mean)**2
    standard_forced_deviation = (forced_deviation / len(row))**0.5
    return standard_forced_deviation

def generate_entropy_distance(this_actual_csv_path, this_entropy_csv_path, nodes):
    import csv
    import antropy as ant

    df = pd.read_csv(this_actual_csv_path, index_col='node')

    # Node ids
    node_ids = df.index.tolist()

    # Open CSV and make writer
    this_entropy_predict_csv = open(this_entropy_csv_path, mode = 'w')
    this_entropy_predict_csv_writer = csv.writer(this_entropy_predict_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Write the header
    this_entropy_predict_csv_writer.writerow(['node', 'distance', 'deviation', 'entropy', 'Target'])

    i = 0
    for index, row in df.iterrows():
        distance = nodes[i].distance_to_endpoint
        deviation_from_initial_energy = cf.INITIAL_ENERGY - forced_deviation(row.tolist(), cf.INITIAL_ENERGY) # the higher the more diseased
        entropy = ant.sample_entropy(row.tolist())
        health_status = 'diseased'
        if entropy < cf.ARIMA_ENTROPY_THRESHOLD and deviation_from_initial_energy > 0.6 * cf.INITIAL_ENERGY:
            health_status = 'healthy'
        this_entropy_predict_csv_writer.writerow([node_ids[i]] + [str(distance)]  + [str(deviation_from_initial_energy)] + [str(entropy)] + [health_status])
        i = i + 1
    del i
    
    this_entropy_predict_csv.close()


def consolidate_errors():
    from datetime import datetime
    tens = list(range(10, 11, 10))

    for ten in tens:
        if ten == 11:
            continue
        # errors = make_segmented_predictions('C:/Users/sanis/Desktop/sdwsn-new/results/2021-09-18/12-35/remaining_energies/MLC/arima/aggregate/actual_remaining_energies.csv', f'C:/Users/sanis/Desktop/sdwsn-new/errors_consolidate/arima_forecasts/{cf.ARIMA_FORECAST_LENGTH}p{cf.ARIMA_FORECAST_LENGTH}-{datetime.now().strftime(r"%Y-%m-%d--%H-%M-%S")}.csv', cf.ARIMA_FORECAST_LENGTH, consolidating_errors=True)
        # errors = make_segmented_predictions('errors_consolidate/datasets/actual_remaining_energies-2000-21-09-2021.csv', f'errors_consolidate/arima_forecasts/{ten}p{ten}-{datetime.now().strftime(r"%Y-%m-%d--%H-%M-%S")}.csv', ten, consolidating_errors=True)
        errors = make_segmented_predictions(r'C:/Users/ENGR. B.K. NUHU/Desktop/IMPLEMENTATIONS/sdwsn-new-arima/results/2023-01-29/17-36/remaining_energies/MLC/arima/actual_remaining_energies.csv', f'errors_consolidate/arima_forecasts/{ten}p{ten}-{datetime.now().strftime(r"%Y-%m-%d--%H-%M-%S")}.csv', ten, consolidating_errors=True)
        errors_dict = OrderedDict({"mae": errors[0], "mse": errors[1]})
        # ec.json_orchestrate(cf.ARIMA_FORECAST_LENGTH, cf.ARIMA_FORECAST_LENGTH, 'arima', errors_dict)
        ec.json_orchestrate(ten, ten, 'arima', errors_dict)

    # ten = 990
    # errors = make_segmented_predictions('errors_consolidate/datasets/actual_remaining_energies-2000-21-09-2021.csv', f'errors_consolidate/arima_forecasts/{ten}p{ten}-{datetime.now().strftime(r"%Y-%m-%d--%H-%M-%S")}.csv', ten, consolidating_errors=True)
    # errors_dict = OrderedDict({"mae": errors[0], "mse": errors[1]})
    # # ec.json_orchestrate(cf.ARIMA_FORECAST_LENGTH, cf.ARIMA_FORECAST_LENGTH, 'arima', errors_dict)
    # ec.json_orchestrate(ten, ten, 'arima', errors_dict)

    


if __name__== "__main__":
    consolidate_errors()
