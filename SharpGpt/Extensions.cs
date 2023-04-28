namespace SharpGpt;

public static class Extensions
{
    public static T[] Slice<T>(this T[] source, int index, int length)
    {       
        T[] slice = new T[length];
        Array.Copy(source, index, slice, 0, length);
        return slice;
    }
    
    public static void Deconstruct<T>(this T[]? array, out T item1, out T item2, out T item3) {
        item1 = default!;
        item2 = default!;
        item3 = default!;
        
        if (array == null)
        {
            return;
        }

        if (array.Length > 2) 
        {
            item3 = array[2];
        }
        
        if (array.Length > 1) 
        {
            item2 = array[1];
        }
        
        if (array.Length > 0) 
        {
            item1 = array[0];
        }
    }
    
    public static void Mutate<T>(this T[] source, Func<T, T> action)
    {
        for (int i = 0; i < source.Length; i++)
        {
            source[i] = action(source[i]);
        }
    }
    
    public static void Mutate<T>(this IList<T> source, Func<T, T> action)
    {
        for (int i = 0; i < source.Count; i++)
        {
            source[i] = action(source[i]);
        }
    }
}