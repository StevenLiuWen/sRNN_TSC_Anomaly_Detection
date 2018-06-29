import tensorflow as tf

class base(object):
    def __init__(self, input, label, config):
        self.endpoints = {}
        self.all_parameters = {}
        self.label = label
        self.config = config
        self.output = None
        self.optimized_parameters = None
        self.losses = None
        self.regularization_loss = None
        self.total_loss = None
        self.input = input
        self.summary_writer = None
        self.all_summary = None
        self._setup()

    def _head(self, input):
        '''
        :param input: the input tensor
        :return: head_output: an output tensor
                 head_parameters: a list of parameters
        '''
        raise Exception('Not Implemented')

    def _body(self, input):
        '''
        :param input: the input tensor
        :return body_output: an output tensor
        :return body_parameters: a list of parameters
        '''
        raise Exception('Not Implemented')

    def _tail(self, input):
        '''
        :param input: the input tensor
        :return tail_output: an output tensor
        :return tail_parameters: a list of parameters
        '''
        raise Exception('Not Implemented')

    def _get_losses(self, pred, label):
        """
        :param pred: the prediction tensor
        :param label: the gound truth tensor
        :return losses: a list of losses
        """
        raise Exception('Not Implemented')

    def _get_testing_loss(self, pred, label):
        """
        :param pred: the prediction tensor
        :param label: the gound truth tensor
        :return loss: a loss tensor
        """
        return None

    def _get_regularization_loss(self, optimized_parameters):
        """
        :param optimized_parameters: parameters that should be optimized
        :return regularization_loss: a regularization loss of optimized parameters
        """
        regularization_loss = []
        for p in optimized_parameters:
            regularization_loss.append(tf.nn.l2_loss(p))
        regularization_loss = tf.add_n(regularization_loss) * self.config['weight_decay']
        return regularization_loss

    def _get_optimized_parameters(self, all_parameters):
        """
        :param all_parameters: all parameters
        :return optimized_parameters: a list of parameters that should be optimized
        """
        raise Exception('Not Implemented')

    def _get_summaries(self, losses, statistics):
        """
            define some summaries using tf.summary.scalar()
        """
        for name, loss in losses.items():
            tf.summary.scalar(name, loss)

        for name, statistic in statistics.items():
            tf.summary.scalar(name, statistic)

    def _get_statistics(self):
        """
        :return: a list of some statistic tensors which can be visualized
        """
        raise Exception('Not Implemented')

    def _setup(self):
        # setup the whole network
        head_output, head_parameters = self._head(self.input)
        body_output, body_parameters = self._body(head_output)
        tail_output, tail_parameters = self._tail(body_output)
        self.endpoints.update({'head_output': head_output,
                               'body_output': body_output,
                               'tail_output': tail_output})

        self.all_parameters.update({'head_parameters': head_parameters,
                                    'body_parameters': body_parameters,
                                    'tail_parameters': tail_parameters})
        self.optimized_parameters = self._get_optimized_parameters(self.all_parameters)

        # setup losses
        self.regularization_loss = self._get_regularization_loss(self.optimized_parameters)
        self.losses = self._get_losses(tail_output, self.label)
        self.total_loss = tf.add_n([self.regularization_loss] + list(self.losses.values()))
        self.testing_loss = self._get_testing_loss(tail_output, self.label)

        # setup statistics
        self.statistics = self._get_statistics()

        # setup summary
        self._get_summaries(self.losses, self.statistics)
        self.summaries = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.config['log_path'])

    def train(self):
        raise Exception('Not Implemented')

    def test(self):
        raise Exception('Not Implemented')
