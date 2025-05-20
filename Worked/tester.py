# import matplotlib.pyplot
# import numpy as np
# import math

# # colors = ['b', 'r', 'k', 'y', 'g', 'c', 'm']
# # list_packet_losses = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
list_packet_losses = [list(range(1, 4001)) for x in range(3)]
# # for i, packetloss in enumerate(list_packet_losses):
# #     tens_split = np.array_split(packetloss, math.ceil(len(packetloss)/10))
# #     averages = np.mean(tens_split, axis=1)
# #     # rounds = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
# #     rounds = [str(10 + x * 10) for x in range(len(averages))]
# #     x = matplotlib.pyplot
# #     fig = x.figure()
# #     ax = fig.add_axes([0,0,1,1])
# #     # ax.set_title('Scenario ' + str(i + 1))
# #     # ax.set_xlabel('tens')
# #     # ax.set_ylabel('avg packet')
# #     ax.bar(rounds,averages, color=colors[i])
# #     # x.title('title', fontsize = 14, fontweight ='bold', color='b')
# #     x.show()


# # try:
# #     print('llllllllllllllllllll', len(packetloss))

# # except (np.AxisError):
# #     print('cf.MAX_ROUNDS must be multiple of 100')


# colors = ['b', 'r', 'k', 'y', 'g', 'c', 'm']
# x = matplotlib.pyplot
# for i, packetloss in enumerate(list_packet_losses):
#     huns_split = np.array_split(packetloss, math.ceil(len(packetloss)/100))
#     averages = np.mean(huns_split, axis=1)

#     rounds = [str(10 + x * 10) for x in range(len(averages))]
#     fig = x.figure()
#     ax = fig.add_axes([0,0,1,1])
#     # ax.set_title('Scenario ' + str(i + 1))
#     # ax.set_xlabel('tens')
#     # ax.set_ylabel('avg packet')
#     ax.bar(rounds,averages, color=colors[i])
#     # x.title('title', fontsize = 14, fontweight ='bold', color='b')
# x.show()
#     print(huns_split)
#     print('--------------------------------------------------------------------------------------------------------------------------------------')
#     print(averages.tolist())
#     print('len of averages: ', len(averages))
#     # break

# # print(list_packet_losses)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Bring some raw data.
# frequencies = [6, 16, 75, 160, 244, 260, 145, 73, 16, 4, 1]
# # In my original code I create a series and run on that, 
# # so for consistency I create a series from the list.
# freq_series = pd.Series(frequencies)


# # Plot the figure.
# plt.figure(figsize=(12, 8))
# ax = freq_series.plot(kind='bar')
# ax.set_title('Amount Frequency')
# ax.set_xlabel('average per hundred')
# ax.set_ylabel('average packet loss')

# rects = ax.patches
# x_labels = ["%d" % int((i+1)*100) for i in range(len(rects))]
# ax.set_xticklabels(x_labels)

# # Make some labels.

# # for rect, label in zip(rects, x_labels):
# #     height = rect.get_height()
# #     ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
# #             ha='center', va='bottom')

# plt.show()











# import matplotlib.pyplot as plt
# from matplotlib.dates import date2num
# import datetime

# x = [
#     datetime.datetime(2011, 1, 4, 0, 0),
#     datetime.datetime(2011, 1, 5, 0, 0),
#     datetime.datetime(2011, 1, 6, 0, 0)
# ]
# x = date2num(x)

# y = [4, 9, 2]
# z = [1, 2, 3]
# k = [11, 12, 13]

# ax = plt.subplot(111)
# ax.bar(x-0.2, y, width=0.2, color='b', align='center')
# ax.bar(x, z, width=0.2, color='g', align='center')
# ax.bar(x+0.2, k, width=0.2, color='r', align='center')
# ax.xaxis_date()

# plt.show()