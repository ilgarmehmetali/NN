package net.milgar.nn;

import java.util.Arrays;
import java.util.List;

public class App {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		float[][][] dataXOR = { { { 0, 0 }, { 0 } }, { { 0, 1 }, { 1 } }, { { 1, 0 }, { 1 } }, { { 1, 1 }, { 0 } } };

		float[][][] dataOR = { { { 0, 0 }, { 0 } }, { { 0, 1 }, { 1 } }, { { 1, 0 }, { 1 } }, { { 1, 1 }, { 1 } } };

		float[][][] dataAND = { { { 0, 0 }, { 0 } }, { { 0, 1 }, { 0 } }, { { 1, 0 }, { 0 } }, { { 1, 1 }, { 1 } } };

		float[][][] dataFullAdder = { { { 0, 0, 0 }, { 0, 0 } }, { { 0, 0, 1 }, { 1, 0 } }, { { 0, 1, 0 }, { 1, 0 } },
				{ { 0, 1, 1 }, { 0, 1 } }, { { 1, 0, 0 }, { 1, 0 } }, { { 1, 0, 1 }, { 0, 1 } },
				{ { 1, 1, 0 }, { 0, 1 } }, { { 1, 1, 1 }, { 1, 1 } } };

		float[][][] data = dataXOR;

		Network network = new Network(data[0][0].length, data[0][0].length, data[0][1].length);

		for (int i = 0; i < 100000; i++) {
			for (float[][] d : data) {
				network.train(d[0], d[1]);
			}
		}

		// System.out.println();
		for (int i = 0; i < 100; i++) {
			for (int j = 0; j < data.length; j++) {
				List<Float> ans = network.feedForward(data[j][0]);
				if ((data[j][1][0] - ans.get(0)) > 0.01f) {
					System.out.format("%s => %s %n", Arrays.toString(data[j][0]),
							Arrays.toString(ans.toArray(new Float[ans.size()])));
				}
			}
		}
	}

}
