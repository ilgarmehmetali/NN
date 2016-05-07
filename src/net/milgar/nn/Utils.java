package net.milgar.nn;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Utils {
	@SuppressWarnings("unchecked")
	public static <T> T as(Object o, Class<T> tClass) {
		return tClass.isInstance(o) ? (T) o : null;
	}

	public static void loadTrainingData(Path path, List<float[]> input, List<float[]> output, int outputCount)
			throws IOException {
		List<String> lines = Files.readAllLines(path);
		Collections.shuffle(lines, new Random(System.nanoTime()));
		for (int i = 0; i < lines.size(); i++) {
			String[] pieces = lines.get(i).split("\\s");
			float[] tmpInput = new float[pieces.length - outputCount];
			float[] tmpOutput = new float[outputCount];
			for (int k = 0; k < pieces.length - outputCount; k++) {
				tmpInput[k] = Float.parseFloat(pieces[k]);
			}
			for (int k = 0; k < outputCount; k++) {
				tmpOutput[k] = Float.parseFloat(pieces[k + (pieces.length - outputCount)]);
			}
			input.add(tmpInput);
			output.add(tmpOutput);
		}
	}

	public static void write(String filename, float[][] x) throws IOException {
		BufferedWriter outputWriter = null;
		outputWriter = new BufferedWriter(new FileWriter(filename));
		for (int i = 0; i < x.length; i++) {
			outputWriter.write(Float.toString(Math.abs(x[i][0])) + " " + Float.toString(Math.abs(x[i][1])) + " "
					+ Float.toString(Math.abs(x[i][2])));
			outputWriter.newLine();
		}
		outputWriter.flush();
		outputWriter.close();
	}
}
