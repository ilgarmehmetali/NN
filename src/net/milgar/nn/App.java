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

		int iterationCount = 200;
		List<String> errors = new ArrayList<>();
		try {
			Path path = FileSystems.getDefault().getPath(".", "data.dat");
			List<float[]> dataInput = new ArrayList<>();
			List<float[]> dataOutput = new ArrayList<>();
			Utils.loadTrainingData(path, dataInput, dataOutput, 2);

			float[][] input = dataInput.toArray(new float[dataInput.size()][]);
			float[][] output = dataOutput.toArray(new float[dataInput.size()][]);
			int trainingSetCount = (int) (input.length * 0.8f);

			Network network = new Network(input[0].length, input[0].length, input[0].length, output[0].length);

			for (int i = 0; i < iterationCount; i++) {
				for (int k = 0; k < trainingSetCount; k++) {
					float[] er = network.train(input[k], output[k]);
					if (k == 0) {
						float mean = 0;
						for (float f : er)
							mean += Math.abs(f);
						mean /= er.length;
						errors.add("" + mean);
					}
				}

				if (i % 100 == 0) {
					for (int j = 0; j < input.length; j++) {
						List<Float> ans = network.feedForward(input[j]);
						// System.out.format("%s => %s %n",
						// Arrays.toString(input[j]),
						// Arrays.toString(ans.toArray(new Float[ans.size()])));
					}
					System.out.println();
				}

			}

			for (int i = 0; i < 1; i++) {
				for (int j = trainingSetCount; j < input.length; j++) {
					List<Float> ans = network.feedForward(input[j]);
					System.out.format("Girdi: %s , Gerçek Çıktı: %s , Modelin Çıktısı: %s %n",
							Arrays.toString(input[j]), Arrays.toString(output[j]),
							Arrays.toString(ans.toArray(new Float[ans.size()])));
				}
			}

			List<String> testResult = new ArrayList<>();
			for (int j = trainingSetCount; j < input.length; j++) {
				List<Float> ans = network.feedForward(input[j]);

				String tmp = "";
				for (float f : output[j])
					tmp += f + " ";

				for (Float f : ans)
					tmp += f + " ";

				testResult.add(tmp);
			}

			try {
				Utils.write("output_mine.dat", testResult);
				Utils.write("error_mine.dat", errors);
			} catch (IOException ex) {
				ex.printStackTrace();
			}

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
