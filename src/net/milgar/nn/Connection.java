package net.milgar.nn;

import java.util.Random;

public class Connection {

	private Neuron from;
	private Neuron to;
	private float weight;

	public Connection(Neuron from, Neuron to) {
		this.from = from;
		this.to = to;
		this.weight = ((float) (new Random()).nextDouble() * 2) - 1;
	}

	public Connection(Neuron from, Neuron to, float weight) {
		this.from = from;
		this.to = to;
		this.weight = weight;
	}

	public Neuron getFrom() {
		return from;
	}

	public void setFrom(Neuron from) {
		this.from = from;
	}

	public Neuron getTo() {
		return to;
	}

	public void setTo(Neuron to) {
		this.to = to;
	}

	public float getWeight() {
		return weight;
	}

	public void setWeight(float weight) {
		this.weight = weight;
	}

	public void adjustWeight(float deltaWeight) {
		this.weight += deltaWeight;
	}
}
