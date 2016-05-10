package net.milgar.nn;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Array;
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

	public static <T> void write(String filename, List<T> x) throws IOException {
		BufferedWriter outputWriter = null;
		outputWriter = new BufferedWriter(new FileWriter(filename));
		for (int i = 0; i < x.size(); i++) {
			outputWriter.write("" + x.get(i));
			outputWriter.newLine();
		}
		outputWriter.flush();
		outputWriter.close();
	}

	public static <T> T[] concatenate(T[] a, T[] b) {
		int aLen = a.length;
		int bLen = b.length;

		@SuppressWarnings("unchecked")
		T[] c = (T[]) Array.newInstance(a.getClass().getComponentType(), aLen + bLen);
		System.arraycopy(a, 0, c, 0, aLen);
		System.arraycopy(b, 0, c, aLen, bLen);

		return c;
	}
}
