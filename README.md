# SA Fakenews

This repository contains the source code for an Honours thesis that aimed to improve disinformation detection.

## Getting Started

To train the models, run one of the following files:
- `Bert_main.py`
- `LSTM_main.py`

The results (accuracy, precision, recall, and F1 scores) on the testing data will be stored in the respective `MODEL_NAME_Results` folder.
Each trained model will be saved in the corresponding `MODEL_NAME_models` folder.

## Project Structure

- The `pipeline.py` file contains the functions used for text cleaning and preprocessing.
- The `models.py` file includes the functions that construct the LSTM and BERT models for training.
- To evaluate the models on a US dataset, navigate to the `USA Testing` folder and run the Jupyter notebook `fake_real.ipynb`.
- The `Whatsapp Bot` folder houses the Flask server code necessary for connecting to the Meta developer dashboard.

## Data Analysis

In the `data analysis` folder, you'll find a Jupyter notebook for exploratory data analysis. Please note that the WordCloud Python package may be experiencing issues. If you encounter problems, you can safely remove the WordCloud import from the notebook and proceed with the remaining analysis.

## Prerequisites

To test the WhatsApp Bot locally, we recommend using [ngrok](https://ngrok.com/download). Follow this helpful [tutorial](https://ngrok.com/docs/integrations/facebook/webhooks/) to connect the ngrok server to the Meta developers dashboard.

## Installation
`keras-nlp` currently only supports Linux, so it's recommended to use a Linux distribution of your choice.

```
pip install -r requirements.txt
```

## Authors

- **Raymond Mugandiwa** - [GitHub Profile](https://github.com/RaymondMugandiwa)
- if you expirence anny issue you can contact me with the email mugandiwaraymond@gmail.com

## License

This project is licensed under the [MIT License](LICENSE.md) - see the LICENSE.md file for details.

## Acknowledgments

- 