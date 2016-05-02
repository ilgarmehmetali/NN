package net.milgar.nn;

import java.util.ArrayList;
import java.util.List;

public class Network {

	private Neuron[][] layers;

	private static final float LEARNING_CONSTANT = 0.5f;

	public Network(int... network) {
		if (network.length < 2) return; // Too few arguments

		layers = new Neuron[network.length][];
		layers[0] = new InputNeuron[network[0] + 1];
		for (int i = 1; i < network.length - 1; i++) {
			layers[i] = new HiddenNeuron[network[i] + 1];
		}
		layers[network.length - 1] = new OutputNeuron[network[network.length - 1]];

		for (int i = 0; i < this.layers[0].length - 1; i++) {
			this.layers[0][i] = new InputNeuron();
		}

		for (int i = 1; i < this.layers.length - 1; i++) {
			for (int j = 0; j < this.layers[i].length - 1; j++) {
				this.layers[i][j] = new HiddenNeuron();
			}
		}

		for (int i = 0; i < this.layers[network.length - 1].length; i++) {
			this.layers[network.length - 1][i] = new OutputNeuron();
		}

		// bias neurons
		this.layers[0][this.layers[0].length - 1] = new InputNeuron(1);

		for (int i = 1; i < this.layers.length - 1; i++) {
			this.layers[i][this.layers[i].length - 1] = new HiddenNeuron(1);
		}

		for (int i = 0; i < this.layers.length - 1; i++) {
			for (Neuron n1 : layers[i]) {
				for (Neuron n2 : layers[i + 1]) {
					Connection c = new Connection(n1, n2);
					n1.addConnection(c);
					n2.addConnection(c);
				}
			}
		}
	}

	public List<Float> feedForward(float[] input) {
		for (int i = 0; i < this.layers[0].length - 1; i++) {
			Neuron[] l = this.layers[0];
			InputNeuron n = (InputNeuron) l[i];
			n.setInput(input[i]);
		}

		for (int i = 1; i < this.layers.length; i++) {
			for (Neuron n : this.layers[i]) {
				n.calcOutput();
			}
		}

		List<Float> answer = new ArrayList<>();

		for (int i = 0; i < this.layers[this.layers.length - 1].length; i++) {
			answer.add(this.layers[this.layers.length - 1][i].getOutput());
		}
		return answer;
	}

	public float train(float[] input, float[] answer) {
		this.feedForward(input);
		float[][] deltaOutputs = new float[this.layers.length][];
		for (int i = 0; i < this.layers.length; i++)
			deltaOutputs[i] = new float[this.layers[i].length];

		for (int i = 0; i < this.layers[this.layers.length - 1].length; i++) {
			Neuron n = this.layers[this.layers.length - 1][i];
			float delta = n.getOutput() * (1 - n.getOutput()) * (answer[i] - n.getOutput());
			n.setDelta(delta);

			for (Connection c : n.getConnections()) {
				float deltaWeight = c.getFrom().getOutput() * delta;
				c.adjustWeight(LEARNING_CONSTANT * deltaWeight);
			}
		}

		for (int i = this.layers.length - 2; i >= 0; i--) {
			for (Neuron hidden : this.layers[i]) {
				float sum = 0f;
				for (Connection c : hidden.getConnections()) {
					if (c.getFrom() == hidden) {
						sum += c.getWeight() * c.getTo().getDelta(); // TODO: her
						// output için
						// deltaOutput önceden hesaplanmalı
					}
				}

				for (Connection c : hidden.getConnections()) {
					if (c.getTo() == hidden) {
						float deltaHiddenOutput = hidden.getOutput() * (1 - hidden.getOutput()) * sum;
						c.getTo().setDelta(deltaHiddenOutput);
						float deltaWeight = c.getFrom().getOutput() * deltaHiddenOutput;
						c.adjustWeight(LEARNING_CONSTANT * deltaWeight);
					}
				}
			}
		}

		/*		
				for (HiddenNeuron[] hiddenNeurons : this.hiddenLayers) {
					for (HiddenNeuron hidden : hiddenNeurons) {
						float sum = 0f;
						for (Connection c : hidden.getConnections()) {
							if (c.getFrom() == hidden) {
								sum += c.getWeight() * deltaOutput; // TODO: her
																	// output için
																	// deltaOutput önceden hesaplanmalı
							}
						}
		
						for (Connection c : hidden.getConnections()) {
							if (c.getTo() == hidden) {
								float deltaHiddenOutput = hidden.getOutput() * (1 - hidden.getOutput()) * sum;
								float deltaWeight = c.getFrom().getOutput() * deltaHiddenOutput;
								c.adjustWeight(LEARNING_CONSTANT * deltaWeight);
							}
						}
		
					}
				}
		*/
		float avgError = 0f;
		for (int i = 0; i < this.layers[this.layers.length - 1].length; i++) {
			avgError += answer[i] - this.layers[this.layers.length - 1][i].getOutput();
		}
		return avgError / this.layers[this.layers.length - 1].length;
	}
}
