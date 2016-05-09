package net.milgar.test;

import java.util.Arrays;
import java.util.List;

import net.milgar.nn.Network;

public class FullAdder {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		int iterationCount = 100000;
		float errors[][] = new float[iterationCount][];

		float[][] input = { { 0, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, { 0, 1, 1 }, { 1, 0, 0 }, { 1, 0, 1 }, { 1, 1, 0 },
				{ 1, 1, 1 } };

		float[][] output = { { 0, 0 }, { 1, 0 }, { 1, 0 }, { 0, 1 }, { 1, 0 }, { 0, 1 }, { 0, 1 }, { 1, 1 } };

		Network network = new Network(input[0].length, input[0].length, output[0].length);

		for (int i = 0; i < iterationCount; i++) {
			for (int k = 0; k < input.length; k++) {
				errors[i] = network.train(input[k], output[k]);
			}

			if (i % 1000 == 0) {
				for (int j = 0; j < input.length; j++) {
					List<Float> ans = network.feedForward(input[j]);
					System.out.format("%s => %s %n", Arrays.toString(input[j]),
							Arrays.toString(ans.toArray(new Float[ans.size()])));
				}
				System.out.println();
			}
		}

		for (int i = 0; i < 1; i++) {
			for (int j = 0; j < input.length; j++) {
				List<Float> ans = network.feedForward(input[j]);
				System.out.format("%s => %s %n", Arrays.toString(output[j]),
						Arrays.toString(ans.toArray(new Float[ans.size()])));
			}
		}

	}

}
