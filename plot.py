from support import bar_chart
import matplotlib.pyplot as plt

fig, ax = bar_chart(title='1x320 unidir GRU network with cross entropy loss', selection=[1, 2, 3])
# fig.savefig("results/bars.pdf", bbox_inches='tight')
fig.savefig("results/bars/bars_1x320GRU_ce.png", bbox_inches='tight')

fig, ax = bar_chart(title='4x320 bidir LSTM network with cross entropy loss', selection=[4, 5, 6])
# fig.savefig("results/bars.pdf", bbox_inches='tight')
fig.savefig("results/bars/bars_4x320LSTM_ce.png", bbox_inches='tight')

fig, ax = bar_chart(title='4x320 bidir LSTM network with CTC loss', selection=[7, 8, 9])
# fig.savefig("results/bars.pdf", bbox_inches='tight')
fig.savefig("results/bars/bars_4x320LSTM_ctc.png", bbox_inches='tight')

fig, ax = bar_chart(title='1x320 LSTM network with with cross entropy loss', selection=[10, 11, 12, 13, 14])
# fig.savefig("results/bars.pdf", bbox_inches='tight')
fig.savefig("results/bars/bars_1x320LSTM_ce.png", bbox_inches='tight')