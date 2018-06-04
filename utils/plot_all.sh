unzip -o results/results_100.zip -d results/
python main/framework_comparison/plot.py
mv results/framework_comparison/1x320-LSTM_cross-entropy.pdf results/framework_comparison/1x320-LSTM_cross-entropy_100.pdf
mv results/framework_comparison/1x320-LSTM_cross-entropy.png results/framework_comparison/1x320-LSTM_cross-entropy_100.png

python main/pytorch_comparison/plot.py
mv results/pytorch_comparison/1x320-LSTM_cross-entropy.pdf results/pytorch_comparison/1x320-LSTM_cross-entropy_100.pdf
mv results/pytorch_comparison/1x320-LSTM_cross-entropy.png results/pytorch_comparison/1x320-LSTM_cross-entropy_100.png

unzip -o results/results_1k.zip -d results/
python main/framework_comparison/plot.py
python main/pytorch_comparison/plot.py
