from data import bar_chart
import matplotlib.pyplot as plt

fig,ax=bar_chart(title='320 unit GRU network with Cross Entropy loss')
fig.savefig("results/bars.pdf", bbox_inches='tight')
plt.show()