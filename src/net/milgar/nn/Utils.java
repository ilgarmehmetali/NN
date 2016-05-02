package net.milgar.nn;

public class Utils {
	public static <T> T as(Object o, Class<T> tClass) {
		return tClass.isInstance(o) ? (T) o : null;
	}
}
