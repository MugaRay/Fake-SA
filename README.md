# SA Fakenews

This repository contains the source code for an Honours thesis that aimed to improve disinformation detection.

## Getting Started

To train the models, run one of the following files:
- `Bert_main.py`
- `LSTM_main.py`

The results (accuracy, precision, recall, and F1 scores) on the testing data will be stored in the respective `MODEL_NAME_Results` folder.
Each trained model will be saved in the corresponding `MODEL_NAME_models` folder.

To test the models on a US dataset, refer to the `USA Testing` folder and the Jupyter notebook `fake_real.ipynb`.

The `Whatsapp Bot` folder contains the Flask server to connect to the Meta developer dashboard.

## Prerequisites

To test the WhatsApp Bot locally, we recommend using [ngrok](https://ngrok.com/download). Follow this helpful [tutorial](https://ngrok.com/docs/integrations/facebook/webhooks/) to connect the ngrok server to the Meta developers dashboard.

## Installation

To set up the project, run the following command:

```
pip install -r requirements.txt
```

## Authors

- **Raymond Mugandiwa** - [GitHub Profile](https://github.com/MugaRay)

## License

This project is licensed under the [MIT License](LICENSE.md) - see the LICENSE.md file for details.

## Acknowledgments

- 
