package net.milgar.nn;

public class InputNeuron extends Neuron {

	public InputNeuron() {
		super();
	}

	public InputNeuron(int bias) {
		super(bias);
	}

	public void setInput(float input) {
		this.output = input;
	}
}
