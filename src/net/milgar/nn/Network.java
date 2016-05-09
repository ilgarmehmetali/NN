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
			InputNeuron n = Utils.as(l[i], InputNeuron.class);
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

	public float[] train(float[] input, float[] answer) {
		this.feedForward(input);

		for (int i = 0; i < this.layers[this.layers.length - 1].length; i++) {
			OutputNeuron n = (OutputNeuron) this.layers[this.layers.length - 1][i];
			n.calcDelta(answer[i] - n.getOutput());
			n.adjustWeights(LEARNING_CONSTANT);
		}

		for (int i = this.layers.length - 2; i >= 0; i--) {
			for (Neuron hidden : this.layers[i]) {
				hidden.calcDelta();
				hidden.adjustWeights(LEARNING_CONSTANT);
			}
		}

		float[] err = new float[this.layers[this.layers.length - 1].length];
		for (int i = 0; i < this.layers[this.layers.length - 1].length; i++) {
			err[i] = this.layers[this.layers.length - 1][i].getOutput() - answer[i];
		}
		return err;
	}
}
