from data import bar_chart
import matplotlib.pyplot as plt

fig,ax=bar_chart(title='4x320 bidir LSTM network with cross entropy loss', selection=[1,2,3])
# fig.savefig("results/bars.pdf", bbox_inches='tight')
fig.savefig("results/bars.png", bbox_inches='tight')
plt.show()