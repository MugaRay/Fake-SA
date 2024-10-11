from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import F1Score, Recall, Precision, AUC
from tensorflow import cast, argmax, float32, int64, reduce_sum
import keras_nlp

def Marivate_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(10000, 128))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Conv1D(filters=128, kernel_size=3, activation='sigmoid'))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(64,return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))
    model.add(keras.layers.Dense(32,activation="tanh",  kernel_regularizer=l2(1e-2)))
    model.add(keras.layers.Dense(32,activation="tanh",  kernel_regularizer=l2(1e-2)))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    f1 = F1Score()
    auc = AUC()
    recall = Recall()
    prec = Precision() 
    model.compile(loss='binary_crossentropy', optimizer=optimizer,  metrics=["accuracy", prec, recall, f1, auc])

    return model


def create_Bert():
    # Load the BERT classifier
    classifier = keras_nlp.models.BertClassifier.from_preset(
        "bert_tiny_en_uncased",
        num_classes=2
    )
    
    
    # Compile the model with custom metrics
    classifier.compile(
        loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.Adam(learning_rate=1e-5),
        metrics=[keras.metrics.SparseCategoricalAccuracy(), F1Score(), Bert_precision, Bert_recall]
    )

    return classifier



def Bert_precision(y_true, y_pred):
    y_pred = argmax(y_pred, axis=1)
    y_true = cast(y_true, int64)
    
    tp = reduce_sum(cast(y_true * y_pred, float32))
    precision = tp / (reduce_sum(cast(y_pred, float32)) + keras.backend.epsilon())
    return precision


def Bert_recall(y_true, y_pred):
    y_true = cast(y_true, float32)
    y_pred = cast(y_pred, float32)
    
    y_pred = argmax(y_pred, axis=1)
    y_true = cast(y_true, int64)
    
    tp = reduce_sum(cast(y_true * y_pred, float32))
    recall = tp / (reduce_sum(cast(y_true, float32)) + keras.backend.epsilon())
    return recall


class F1Score(keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = argmax(y_pred, axis=-1)
        y_true = cast(y_true, int64)
        
        self.true_positives.assign_add(reduce_sum(cast((y_true == 1) & (y_pred == 1), float32)))
        self.false_positives.assign_add(reduce_sum(cast((y_true == 0) & (y_pred == 1), float32)))
        self.false_negatives.assign_add(reduce_sum(cast((y_true == 1) & (y_pred == 0), float32)))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + keras.backend.epsilon())
        return 2 * (precision * recall) / (precision + recall + keras.backend.epsilon())

    def reset_states(self):
        for var in self.variables:
            var.assign(0)

