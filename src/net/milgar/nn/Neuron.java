package net.milgar.nn;

import java.util.ArrayList;
import java.util.List;

public class Neuron {

	protected Float output;
	protected List<Connection> connections;
	protected boolean bias;
	protected float delta;
	protected float sum;

	public Neuron() {
		this.connections = new ArrayList<>();
		this.bias = false;
	}

	public Neuron(int bias) {
		this.connections = new ArrayList<>();
		this.output = new Float(bias);
		this.bias = true;
	}

	public void calcOutput() {
		if (!this.bias) {
			this.sum = 0f;
			for (Connection c : this.connections) {
				if (c.getTo() == this) {
					sum += c.getWeight() * c.getFrom().getOutput();
				}
			}
			this.output = new Float(sigmoid(sum));
		}
	}

	public void calcDelta() {
		float sum = 0f;
		for (Connection c : this.getConnections()) {
			if (c.getFrom() == this) {
				sum += c.getWeight() * c.getTo().getDelta();
			}
		}

		float deltaHiddenOutput = this.getOutput() * (1 - this.getOutput()) * sum;
		this.setDelta(deltaHiddenOutput);
	}

	public void adjustWeights(float LEARNING_CONSTANT) {
		for (Connection c : this.getConnections()) {
			if (c.getTo() == this) {
				float deltaWeight = c.getFrom().getOutput() * this.getDelta();
				c.adjustWeight(LEARNING_CONSTANT * deltaWeight);
			}
		}
	}

	public void addConnection(Connection c) {
		this.connections.add(c);
	}

	public void resetDelta() {
		this.delta = 0;
	}

	public void setDelta(float value) {
		this.delta = value;
	}

	public float getDelta() {
		return this.delta;
	}

	public float getSum() {
		return this.sum;
	}

	public float getOutput() {
		return this.output;
	}

	public static float sigmoid(float x) {
		return 1f / (1f + (float) Math.exp(-x));
	}

	public List<Connection> getConnections() {
		return this.connections;
	}

}
