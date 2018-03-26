#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import iris_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    print('\n----COLETA DAS INFORMAÇÕES E MONTAGEM DO MODELO--')
    # Fetch the data
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()
    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[25, 25, 25],
        # The model must choose between 2 classes.
        n_classes=2)
    
    print('\n-----------------TREINO---------------------------')
    # Train the Model
    classifier.train(
        input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size), steps=args.train_steps)
    
    print('\n-----------------TESTES---------------------------')
    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, args.batch_size))
    
    template_teste = ('\nA precisão no conjunto de teste foi de: {:.3f}%')
    print(template_teste.format(float(eval_result['accuracy'])*100))
    
    # Generate predictions from the model
    predict_x = {
        'TEMPERATURA_MAXIMA': [-20, 25, 30, 35, 40],
        'UMIDADE_RELATIVA': [100, 20, 30, 40, 85],
    }

    predictions = classifier.predict(
        input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    print('\n-----------------PREVISAO---------------------------')
    template = ('\nA previsão é de "{}" ({:.3f}%)')

    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.PREVISAO[class_id],
                              100 * probability))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
