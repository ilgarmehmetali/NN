package net.milgar.nn;

public class OutputNeuron extends Neuron {

	public OutputNeuron() {
		super();
	}

	public void calcDelta() {
		throw new UnsupportedOperationException();
	}

	public void calcDelta(float error) {
		float delta = this.getOutput() * (1 - this.getOutput()) * error;
		this.setDelta(delta);
	}
}
