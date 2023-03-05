import tensorflow as tf


def log_scalar(summary_writer, log_name, log_value, step_or_epoch_value):
    
    with summary_writer.as_default():
        tf.summary.scalar(log_name, log_value, step=step_or_epoch_value)


def log_sequence_of_scalars(summary_writer, log_name, log_value_vector, step_or_epoch_value_vector):

    with summary_writer.as_default():
        for log_value,step_or_epoch_value in zip(log_value_vector, step_or_epoch_value_vector):
            tf.summary.scalar(log_name, log_value, step=step_or_epoch_value)

