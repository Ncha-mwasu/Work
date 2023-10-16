import config as cf

tens = list(range(10, 1001, 10))

for ten in tens:
    cf.MARKOV_PREDICTION_INTERVAL = ten
    import run
    print(ten)

    # for i in range(0, 10):
        # print('This interval: ', cf.MARKOV_PREDICTION_INTERVAL)
    #     cf.MARKOV_PREDICTION_INTERVAL = i
    #     cf.MARKOV_PREDICTION_INTERVAL = i
    #     # should be the same
    #     print('This ten: ', ten)

# import sys
