package net.milgar.nn;

import java.util.ArrayList;
import java.util.List;

public class Neuron {

	protected Float output;
	protected List<Connection> connections;
	protected boolean bias;

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
			float sum = 0f;
			for (Connection c : this.connections) {
				if (c.getTo() == this) {
					sum += c.getWeight() * c.getFrom().getOutput();
				}
			}
			this.output = new Float(sigmoid(sum));
		}
	}

	public void addConnection(Connection c) {
		this.connections.add(c);
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
