# Neural Network from scratch

This repository contains a basic implementation of a neural network from scratch. The purpose of this project is to provide a deeper insight about how neural networks work and can be implemented using basic Python libraries. 
<!-- To build a neural network from scratch on your own, you may follow this article: -->

## Getting Started

To get started with this project, follow these steps:

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/your-username/NeuralNet-Barebones.git
   ```
   
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   or
   ```bash
   conda env create -f environment.yml
   ```

3. **Build a Neural network**:
   - Create a new notebook or python file
   - From model.py import Neural_network and start building model using a tensorflow like API:
     ```bash
     model = NeuralNetwork(
       [
           (784, 'relu'),
           (128, 'relu'),
           (64, 'relu'),
           (10, 'softmax')
       ]
      )

      history = model.fit(
          trainx, trainy, 
          epochs=10, 
          learning_rate=0.1,  
          lossFunction='cross_entropy',
          batch_size=32,
          validation_data = (x_validate, y_validate)
      )
     ```

## Structure

- `model.py`: Contains the main script for building and training neural networks with a tensorflow-like API.
- `nnscratch-detailed.ipynb`: Contains a detailed walkthrough of how to build one yourself.
<!-- - `utils.py`: Contains utility functions for data preprocessing and evaluation. -->


## Contributing

Contributions are welcome! If you have any improvements or want to add new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
