package net.milgar.nn;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class App {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		int iterationCount = 3000;
		float errors[][] = new float[iterationCount][];
		try {
			Path path = FileSystems.getDefault().getPath(".", "data.dat");
			List<float[]> dataInput = new ArrayList<>();
			List<float[]> dataOutput = new ArrayList<>();
			Utils.loadTrainingData(path, dataInput, dataOutput, 3);

			float[][] input = dataInput.toArray(new float[dataInput.size()][]);
			float[][] output = dataOutput.toArray(new float[dataInput.size()][]);
			int trainingSetCount = (int) (input.length * 0.8f);

			// input = dataFullAdderInput;
			// output = dataFullAdderOutput;

			Network network = new Network(input[0].length, input[0].length, input[0].length, output[0].length);

			for (int i = 0; i < iterationCount; i++) {
				for (int k = 0; k < trainingSetCount; k++) {
					errors[i] = network.train(input[k], output[k]);
				}

				if (i % 100 == 0) {
					for (int j = 0; j < input.length; j++) {
						List<Float> ans = network.feedForward(input[j]);
						System.out.format("%s => %s %n", Arrays.toString(input[j]),
								Arrays.toString(ans.toArray(new Float[ans.size()])));
					}
					System.out.println();
				}

			}

			for (int i = 0; i < 1; i++) {
				for (int j = trainingSetCount; j < input.length; j++) {
					List<Float> ans = network.feedForward(input[j]);
					System.out.format("%s => %s %n", Arrays.toString(output[j]),
							Arrays.toString(ans.toArray(new Float[ans.size()])));
				}
			}

			try {
				Utils.write("error.dat", errors);
			} catch (Exception ex) {

			}

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
