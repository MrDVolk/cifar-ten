import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history, title=''):
    colors = ['r', 'k']
    metrics= ['f1_score', 'accuracy']
    
    x = range(1, len(history['accuracy']) + 1)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.ylim(0, 1.1)
    for i, name in enumerate(metrics):
        plt.plot(x, history[name], colors[i], label=name) 
    plt.title(f'Training {title}')
    plt.legend()
        
    plt.subplot(1, 2, 2)
    plt.ylim(0, 1.1)
    for i, name in enumerate(metrics):
        name = f'val_{name}'
        plt.plot(x, history[name], colors[i], label=name)
    plt.title(f'Validation {title}')
    plt.legend()
    
    
def plot_convolutions(nn_model, image_example, take=None, title=None):
    vector = np.array([image_example])
    
    # no point in watching anything that is not convolution or input layer
    printable_layers = ['input', 'conv']
    layers = [layer for layer in nn_model.layers if any([layer.name.startswith(pl_name) for pl_name in printable_layers])]
    layer_names = [layer.name for layer in layers]
    
    visualization_model = tf.keras.Model(nn_model.input, [layer.output for layer in layers])
    visualization_results = visualization_model.predict(vector)
    
    visualizations_zip = zip(layer_names, visualization_results)

    if title is not None:
        plt.title(title)
    plt.imshow(image_example)
    plt.axis('off')
    
    for i, (name, vis_result) in enumerate(visualizations_zip):
        # limit layers by 'take' parameter
        if take is not None and i >= take:
            break 

        vis_result_raveled = vis_result[0]
        # no point in watching 1x1 convolutions just by themselves
        if vis_result_raveled.shape[0] is 1 or vis_result_raveled.shape[1] is 1:
            continue

        dimensions = vis_result_raveled.shape[2]
        columns = 5
        
        import math        
        rows = math.ceil(dimensions / columns)
        fig = plt.figure(figsize=(columns * 8, rows * 8))

        for dim in range(0, dimensions):
            vis_result_dimension_splice = np.take(vis_result_raveled, [dim], axis=2)
            image_shape = vis_result_dimension_splice.shape[0], vis_result_dimension_splice.shape[1]
            image = np.reshape(vis_result_dimension_splice, image_shape)

            ax = fig.add_subplot(rows, columns, dim+1)
            ax.set_title(f'{name}::{dim}')
            plt.imshow(image)
            plt.axis('off')
            

def get_test_results(nn_model, test_data):
    test_data_x, test_data_y = test_data
    
    predictions = nn_model.predict(test_data_x)
    answers = np.argmax(predictions, axis=1)
    correct_answers = np.argmax(test_data_y, axis=1)
    nn_was_correct = answers == correct_answers

    test_results = []
    for i, answer in enumerate(answers):
        row = {
            'index': i,
            'answer': answer,
            'correct_answer': correct_answers[i],
            'correct': nn_was_correct[i],
            'confidence': predictions[i][answer]
        }
        test_results.append(row)

    dataframe = pd.DataFrame(test_results)
    return dataframe


def plot_filters(nn_model, layer_index=1):
    filters, biases = nn_model.layers[layer_index].get_weights()
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    filters_count = filters.shape[3]
    channel_count = filters.shape[2]
    plot_index = 1
    fig = plt.figure(figsize=(8, 16))
    for i in range(0, filters_count):
        single_filter = filters[:, :, :, i]
        for j in range(0, channel_count):
            channel_filter = single_filter[:, :, j]

            ax = plt.subplot(filters_count, channel_count, plot_index)
            plt.imshow(channel_filter, cmap='gray')
            plt.axis('off')
            plot_index += 1