package net.milgar.nn;

import java.util.ArrayList;
import java.util.List;

public class Network {

	private InputNeuron[] inputNeurons;
	private HiddenNeuron[] hiddenNeurons;
	private OutputNeuron[] outputNeurons;

	private static final float LEARNING_CONSTANT = 0.5f;

	public Network(int inputs, int hiddens, int outputs) {
		this.inputNeurons = new InputNeuron[inputs + 1]; // +1 bias
		this.hiddenNeurons = new HiddenNeuron[hiddens + 1]; // +1 bias
		this.outputNeurons = new OutputNeuron[outputs]; // +1 bias

		for (int i = 0; i < this.inputNeurons.length - 1; i++) {
			this.inputNeurons[i] = new InputNeuron();
		}

		for (int i = 0; i < this.hiddenNeurons.length - 1; i++) {
			this.hiddenNeurons[i] = new HiddenNeuron();
		}

		for (int i = 0; i < this.outputNeurons.length; i++) {
			this.outputNeurons[i] = new OutputNeuron();
		}

		// bias neurons
		this.inputNeurons[this.inputNeurons.length - 1] = new InputNeuron(1);
		this.hiddenNeurons[this.hiddenNeurons.length - 1] = new HiddenNeuron(1);

		for (int i = 0; i < this.inputNeurons.length; i++) {
			for (int k = 0; k < this.hiddenNeurons.length; k++) {
				Connection c = new Connection(this.inputNeurons[i], this.hiddenNeurons[k]);
				this.inputNeurons[i].addConnection(c);
				this.hiddenNeurons[k].addConnection(c);
			}
		}

		for (int i = 0; i < this.hiddenNeurons.length; i++) {
			for (int k = 0; k < this.outputNeurons.length; k++) {
				Connection c = new Connection(this.hiddenNeurons[i], this.outputNeurons[k]);
				this.hiddenNeurons[i].addConnection(c);
				this.outputNeurons[k].addConnection(c);
			}
		}
	}

	public List<Float> feedForward(float[] input) {
		for (int i = 0; i < this.inputNeurons.length - 1; i++) {
			this.inputNeurons[i].setInput(input[i]);
		}

		for (Neuron n : this.hiddenNeurons) {
			n.calcOutput();
		}

		for (Neuron n : this.outputNeurons) {
			n.calcOutput();
		}

		List<Float> answer = new ArrayList<>();

		for (int i = 0; i < this.outputNeurons.length; i++) {
			answer.add(this.outputNeurons[i].getOutput());
		}

		return answer;
	}

	public float train(float[] input, float[] answer) {
		this.feedForward(input);
		for (int i = 0; i < this.outputNeurons.length; i++) {
			OutputNeuron n = this.outputNeurons[i];
			float deltaOutput = n.getOutput() * (1 - n.getOutput()) * (answer[i] - n.getOutput());

			for (Connection c : n.getConnections()) {
				float deltaWeight = c.getFrom().getOutput() * deltaOutput;
				c.adjustWeight(LEARNING_CONSTANT * deltaWeight);
			}

			for (HiddenNeuron hidden : this.hiddenNeurons) {
				/*
				 * float hiddenError = 0; float deltaHiddenOutput = 0; for
				 * (Connection c : hidden.getConnections()) { if (c.getTo() ==
				 * n) { hiddenError = deltaOutput * c.getWeight(); break; } }
				 * for (Connection c : hidden.getConnections()) { if (c.getTo()
				 * == hidden) { deltaHiddenOutput = hiddenError *
				 * hidden.getOutput() * (1 - hidden.getOutput()); break; } } for
				 * (Connection c : hidden.getConnections()) { if (c.getTo() ==
				 * hidden) { float deltaWeight = c.getTo().getOutput() *
				 * deltaHiddenOutput; c.adjustWeight(LEARNING_CONSTANT *
				 * deltaWeight); } }
				 */

				float sum = 0f;
				for (Connection c : hidden.getConnections()) {
					if (c.getFrom() == hidden) {
						sum += c.getWeight() * deltaOutput;
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

		float avgError = 0f;
		for (int i = 0; i < this.outputNeurons.length; i++) {
			avgError += answer[i] - this.outputNeurons[i].getOutput();
		}
		return avgError / this.outputNeurons.length;
	}
}
