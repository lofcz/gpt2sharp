namespace SharpGpt;

public static class MathExt
{
    public const float Epsilon = 1e-5f;
    public const float Mask = -1e10f;
    public static readonly float Sqrt2OverPi = (float)Math.Sqrt(2 / Math.PI);
    public const float GeluConst = 0.044715f;
    public static Random random = Random.Shared;

    public static void AddMatrix(this ArraySegment<float> dest, ArraySegment<float> a, IList<float> b)
    {
        for (int i = 0; i < a.Count; i++)
        {
            dest[i] = a[i] + b[i];
        }
    }

    public static Model.Tensor<float> CausalMask(this int dim)
    {
        int dim2 = dim * dim;
        Model.Tensor<float> mask = new Model.Tensor<float>(new float[dim2], new[] { dim, dim });

        for (int i = 0; i < dim2; i++)
        {
            mask.Data[i] = i % dim >= i / dim + 1 ? Mask : 0;
        }

        return mask;
    }

    public static float Variance(this IList<float> data, float n)
    {
        return (float)data.Sum(x => Math.Pow(x - n, 2)) / data.Count;
    }

    public static Model.Tensor<float> Norm(this Model.Tensor<float> source, Model.Tensor<float> gain, Model.Tensor<float> bias)
    {
        int last = source.Shape.Last();
        int count = source.Data.Length / last;
        Model.Tensor<float> ret = source.Duplicate();

        for (int i = 0; i < count; i++)
        {
            ArraySegment<float> layer = ret.Slice(i * last, last);
            float mean = layer.Average();
            float variance = layer.Variance(mean);

            for (int j = 0; j < layer.Count; j++)
            {
                layer[j] = gain.Data[j] * (layer[j] - mean) / (float)Math.Sqrt(variance + Epsilon) + bias.Data[j];
            }
        }

        return ret;
    }

    public static Model.Tensor<float> MutateSoftmax(this Model.Tensor<float> a)
    {
        int dim = a.Shape.LastOrDefault();
        int n = a.Data.Length / dim;

        for (int i = 0; i < n; i++)
        {
            ArraySegment<float> layer = new ArraySegment<float>(a.Data, i * dim, dim);
            float max = layer.Max();
            layer.Mutate(x => (float)Math.Exp(x - max));
            float sum = layer.Sum();
            layer.Mutate(x => x / sum);
        }

        return a;
    }

    public static Model.Tensor<float> Merge(this List<Model.Tensor<float>> a, int size)
    {
        int dA = a.Any() ? a[0].Shape[0] : 0;
        int mergedSize = dA * size;
        Model.Tensor<float> merged = new Model.Tensor<float>(new float[mergedSize], new [] { dA, size });
        int chunkSize = a[0].Shape.LastOrDefault();
        int[] offsets = new int[a.Count];
        int outOffset = 0;
        int chunks = mergedSize / size;

        for (int i = 0; i < chunks; i++)
        {
            for (int j = 0; j < a.Count; j++)
            {
                Array.Copy(a[j].Data, offsets[j], merged.Data, outOffset, chunkSize);
                offsets[j] += chunkSize;
                outOffset += chunkSize;
            }
        }

        return merged;
    }

    public static Model.Tensor<float> MutateAdd(this Model.Tensor<float> a, Model.Tensor<float> b)
    {
        for (int i = 0; i < a.Data.Length; i++)
        {
            a.Data[i] += b.Data[i];
        }

        return a;
    }
    
    public static Model.Tensor<float> Multiply(this Model.Tensor<float> a, Model.Tensor<float> b)
    {
        Model.Tensor<float> ret = new Model.Tensor<float>(new float[a.Shape[0] * b.Shape[1]], new[] { a.Shape[0], b.Shape[1] });

        int a1 = a.Shape[1];
        int b1 = b.Shape[1];
       
        Parallel.For(0, a.Shape[0], i =>
        {
            for (int j = 0; j < b1; j++) {
                for (int k = 0; k < a1; k++) {
                    ret.Data[i * b1 + j] += a.Data[i * a1 + k] * b.Data[k * b1 + j];
                }
            }
        });

        return ret;
    }

    public static Model.Tensor<float> Transpose(this Model.Tensor<float> a)
    {
        int dA = a.Shape[0];
        int dB = a.Shape[1];
        
        Model.Tensor<float> transposed = new Model.Tensor<float>(new float[a.Data.Length], new [] { dB, dA });

        for (int i = 0; i < dA; i++)
        {
            for (int j = 0; j < dB; j++)
            {
                transposed.Data[j * dA + i] = a.Data[i * dB + j];
            }
        }
        
        return transposed;
    }

    public static Model.Tensor<float> MutateGelu(this Model.Tensor<float> a)
    {
        a.Data.Mutate(x => (float)(x * 0.5f * (Math.Tanh(Sqrt2OverPi * (GeluConst * Math.Pow(x, 3) + x)) + 1)));
        return a;
    }

    public static Model.Tensor<float> Linear(this Model.Tensor<float> source, Model.Tensor<float> weight, Model.Tensor<float> bias)
    {
        Model.Tensor<float> multiplied = Multiply(source, weight);

        for (int i = 0; i < multiplied.Shape[0]; i++)
        {
            ArraySegment<float> slice = multiplied.Slice(i);
            slice.AddMatrix(slice, bias.Data);
        }

        return multiplied;
    }

    public static Model.Tensor<float>[] Split(this Model.Tensor<float> source, int chunkSize)
    {
        int chunks = source.Shape.LastOrDefault() / chunkSize;
        int dataPerChunk = source.Data.Length / chunks;
        int parts = source.Data.Length / chunkSize;
        
        Model.Tensor<float>[] ret = new Model.Tensor<float>[chunks];

        for (int i = 0; i < chunks; i++)
        {
            ret[i] = new Model.Tensor<float>(new float[dataPerChunk], new [] { source.Shape[^2], chunkSize });
        }

        int batch = 0;
        for (int i = 0; i < parts; i++)
        {
            int copyTo = i % chunks;
            float[] slice = source.Data.Slice(i * chunkSize, chunkSize);
            
            Array.Copy(slice, 0, ret[i % chunks].Data, batch * chunkSize, chunkSize);
            
            if (copyTo == chunks - 1)
            {
                batch++;
            }
        }

        return ret;
    }
}